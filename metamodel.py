from   typing import Dict, Tuple, List, Optional


import torch
import torch.nn as nn
from   torch.utils.data import TensorDataset, DataLoader

import numpy  as np
import pandas as pd

from   sklearn.linear_model    import LinearRegression


# import losses  # architecture



# ----------------------------------------------------------------------
# linear regression for metamodel
# ----------------------------------------------------------------------

def weights_LR_metamodel(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_valid: Optional[pd.DataFrame] = None,
                      X_test : Optional[pd.DataFrame] = None,
                      verbose: int = 0) \
        -> Tuple[Dict[str, float], pd.Series, pd.Series or None, pd.Series or None]:
    # X: (N, 3), y: (N,)

    model_meta = LinearRegression(fit_intercept=False, positive=True)
    model_meta.fit(X_train, y_train)

    if verbose >= 3:
        pred_train = model_meta.predict(X_train)
        _output = pd.concat([
             y_train, X_train,
             pd.Series(pred_train, name='meta', index=y_train.index)], axis=1)
        _output.columns=['true'] + X_train.columns + ['meta']
        print(_output.astype(np.float32).round(2))

    weights_meta = {name: round(float(coeff), 3) for name, coeff in \
           zip(X_train.columns, model_meta.coef_)}

    pred_train = model_meta.predict(X_train)
    pred_valid = model_meta.predict(X_valid) if X_valid is not None else None
    pred_test  = model_meta.predict(X_test)  if X_test  is not None else None

    # print(f"types: {type(pred_train)}, {type(pred_valid)}, {type(pred_test)}")

    return (weights_meta,
            pd.Series(pred_train, index=X_train.index),
            pd.Series(pred_valid, index=X_valid.index),
            pd.Series(pred_test,  index=X_test .index))





# ----------------------------------------------------------------------
# Metamodel based on a small NN
# ----------------------------------------------------------------------


# 1. META-MODEL ARCHITECTURE
# ============================================================

class MetaNet(nn.Module):
    """
    Meta-model that learns to combine NN, LR, RF predictions.

    Takes context features and outputs weights for each predictor.
    """
    def __init__(self, context_dim: int, num_predictors: int,
                 dropout: float, num_cells = [int, int]):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(context_dim, num_cells[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_cells[0], num_cells[1]),
            nn.ReLU(),
            nn.Linear(num_cells[1], num_predictors)
        )

    def forward(self, context, preds):
        """
        Args:
            context: (B, F) - contextual features (temp, time, etc.)
            preds:   (B, 3) - predictions from [nn, lr, rf]

        Returns:
            y_meta:  (B,)   - weighted combination
            weights: (B, 3) - learned weights
        """
        # Compute weights from context
        logits  = self.net(context)              # (B, 3)
        weights = torch.softmax(logits, dim=-1)  # (B, 3)

        # Weighted combination
        y_meta = (weights * preds).sum(dim=-1)   # (B,)

        return y_meta, weights


# 2. PREPARE DATA
# ============================================================

def prepare_meta_data(
    name: str,
    dict_pred_GW,
    dict_baseline_GW,
    X_features_GW,
    y_true_GW,
    dates,
    feature_cols
):
    """
    Prepare data for meta-model training.

    Returns DataFrame with aligned predictions, features, and targets.
    """
    # Combine all predictions and features
    df = pd.DataFrame({
        'y_true':y_true_GW,
        'nn':    dict_pred_GW['q50'],
        'lr':    dict_baseline_GW.get('lr', np.nan),
        'rf':    dict_baseline_GW.get('rf', np.nan),
    }, index=dates)

    # Add context features
    df_features = pd.DataFrame(X_features_GW, index=dates, columns=feature_cols)
    if 'hour_norm' not in df_features.columns:
        df_features['hour_norm'] = \
                (df_features.index.hour + df_features.index.minute/60) / 24

    df = pd.concat([df, df_features], axis=1)

    # Drop rows with any NaN
    df = df.dropna()

    print(f"Meta-model {name} data: {df.shape}")
    return df


# 3. TRAIN META-MODEL
# ============================================================

# Prepare tensors
def to_tensors(df: pd.DataFrame, feature_cols: List[str]):

    # Predictions
    preds = torch.tensor(
        df[['nn', 'lr', 'rf']].values,
        dtype=torch.float32
    )

    # Context features (exclude predictions and target)

    if 'hour_norm' not in df.columns:
        df['hour_norm'] = (df.index.hour + df.index.minute/60) / 24

    context_cols = [c for c in feature_cols
        if c not in ['consumption_nn', 'consumption_lr', 'consumption_rf']]
    context = torch.tensor(
        df[context_cols].values,
        dtype=torch.float32
    )

    # Target
    y_true = torch.tensor(df['y_true'].values, dtype=torch.float32)

    return preds, context, y_true


def train_meta_model(
    meta_net,
    df_train,
    df_valid,
    feature_cols,
    epochs      : int,
    lr          : float,
    weight_decay: float,
    batch_size  : int,
    patience    : int,
    factor      : float,
    device
):
    """
    Train meta-model on training data, validate on validation data.
    """

    optimizer = torch.optim.Adam(meta_net.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=factor)  #, verbose=True

    preds_train, context_train, y_train = to_tensors(df_train, feature_cols)
    preds_valid, context_valid, y_valid = to_tensors(df_valid, feature_cols)

    # Create datasets and loaders
    train_dataset= TensorDataset(preds_train, context_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,   shuffle=True)

    valid_dataset= TensorDataset(preds_valid, context_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False)

    meta_net = meta_net.to(device)
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        # Training
        meta_net.train()
        train_loss_total = 0.

        for preds_train, context_train, y_train in train_loader:
            preds_train  = preds_train  .to(device)
            context_train= context_train.to(device)
            y_train      = y_train      .to(device)

            optimizer.zero_grad()

            # Forward pass
            y_meta, weights = meta_net(context_train, preds_train)

            # Loss
            loss_train = criterion(y_meta, y_train)

            # Backward
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(meta_net.parameters(), 1.)
            optimizer.step()

            train_loss_total += loss_train.item() * len(y_train)

        train_loss_avg = train_loss_total / len(train_dataset)


        # Validation
        meta_net.eval()
        valid_loss_total = 0.

        with torch.no_grad():
            for preds_valid, context_valid, y_valid in valid_loader:
                preds_valid  = preds_valid  .to(device)
                context_valid= context_valid.to(device)
                y_valid      = y_valid      .to(device)

                y_meta_valid, weights_valid = meta_net(context_valid, preds_valid)
                loss_valid = criterion(y_meta_valid, y_valid)

                valid_loss_total += loss_valid.item() * len(y_valid)

        valid_loss_avg = valid_loss_total / len(valid_dataset)

        # Learning rate scheduling
        scheduler.step(valid_loss_avg)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_weights = weights_valid.mean(dim=0).cpu().numpy()
            print(f"Epoch{epoch+1:3n}/{epochs}: "
                  f"losses train{train_loss_avg:6.2f}, "
                  f"valid{valid_loss_avg:6.2f}; "
                  f"avg weights: NN{avg_weights[0]*100:5.1f}%, "
                  f"LR{avg_weights[1]*100:5.1f}%, RF{avg_weights[2]*100:5.1f}%")

        # Save best model
        if valid_loss_avg < best_valid_loss:
            best_valid_loss = valid_loss_avg
            torch.save(meta_net.state_dict(), 'cache/best_metamodel.pth')

    # Load best model
    meta_net.load_state_dict(torch.load('cache/best_metamodel.pth', weights_only=True))

    return meta_net




def metamodel_NN(list_dict_pred, list_dict_baseline, list_X, list_y, list_dates,
                feature_cols,
                #constants
                dropout   : float,
                num_cells : int,
                epochs    : int,
                lr        : float,
                weight_decay:float,
                patience  : int,
                factor    : float,
                batch_size: int,
                device):

    [dict_pred_train_GW, dict_pred_valid_GW, dict_pred_test_GW] = list_dict_pred
    [dict_baseline_train_GW,dict_baseline_valid_GW,dict_baseline_test_GW] = \
        list_dict_baseline
    [X_train_GW,  X_valid_GW, X_test_GW] = list_X
    [y_train_GW,  y_valid_GW, y_test_GW] = list_y
    [train_dates, valid_dates,test_dates]= list_dates


    _feature_cols = feature_cols + ['hour_norm']

    # Prepare data
    df_meta_train = prepare_meta_data(
        "train",
        dict_pred_train_GW,
        dict_baseline_train_GW,
        X_train_GW,
        y_train_GW,
        train_dates,
        feature_cols  # /!\ 'hour_norm' will be added inside
    )

    df_meta_valid = prepare_meta_data(
        "valid",
        dict_pred_valid_GW,
        dict_baseline_valid_GW,
        X_valid_GW,
        y_valid_GW,
        valid_dates,
        feature_cols  # /!\ 'hour_norm' will be added inside
    )

    # Initialize meta-model
    context_cols = [c for c in _feature_cols
                    if c not in ['consumption_nn', 'consumption_lr', 'consumption_rf']]
    print(f"len feature_cols {len(feature_cols)}, _feature_cols {len(_feature_cols)}, "
          f"context_cols {len(context_cols)}")

    meta_net = MetaNet(context_dim=len(context_cols), num_predictors=3,
                                 dropout=dropout, num_cells=num_cells)

    # Train
    meta_net = train_meta_model(
        meta_net,
        df_meta_train,
        df_meta_valid,
        _feature_cols,
        epochs      ,
        lr          ,
        weight_decay,
        batch_size  ,
        patience    ,
        factor      ,
        device
    )

    # Test
    df_meta_test = prepare_meta_data(
        "test",
        dict_pred_test_GW,
        dict_baseline_test_GW,
        X_test_GW,
        y_test_GW,
        test_dates,
        feature_cols  # /!\ 'hour_norm' will be added inside
    )

    preds_train, context_train, y_train = to_tensors(df_meta_train, _feature_cols)
    preds_valid, context_valid, y_valid = to_tensors(df_meta_valid, _feature_cols)
    preds_test,  context_test,  y_test =  to_tensors(df_meta_test,  _feature_cols)

    meta_net.eval()
    with torch.no_grad():
        pred_meta2_train, weights_train = \
            meta_net(context_train.to(device), preds_train.to(device))
        pred_meta2_valid, weights_valid = \
            meta_net(context_valid.to(device), preds_valid.to(device))
        pred_meta2_test,  weights_test  = \
            meta_net(context_test .to(device), preds_test .to(device))

    # Evaluate
    rmse_test = torch.sqrt(torch.mean((pred_meta2_test - y_test.to(device))**2))
    print(f"\nTest RMSE: {rmse_test.item():.2f} GW")

    # Analyze learned weights
    avg_weights_test = weights_test.mean(dim=0).cpu().numpy()
    print(f"Average test weights: NN={avg_weights_test[0]*100:.1f}%, "
          f"LR={avg_weights_test[1]*100:.1f}%, RF={avg_weights_test[2]*100:.1f}%")


    pred_meta2_train = pd.Series(pred_meta2_train.cpu().numpy(), index=df_meta_train.index)
    pred_meta2_valid = pd.Series(pred_meta2_valid.cpu().numpy(), index=df_meta_valid.index)
    pred_meta2_test  = pd.Series(pred_meta2_test .cpu().numpy(), index=df_meta_test .index)


    return pred_meta2_train, pred_meta2_valid, pred_meta2_test
