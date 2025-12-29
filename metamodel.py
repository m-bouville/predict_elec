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
    if 'horizon' not in df_features.columns:
        df_features['horizon']= (df_features.index.hour*2 + \
                                 df_features.index.minute/30).round().astype(np.int16)

    df = pd.concat([df, df_features], axis=1)

    # Drop rows with any NaN
    df = df.dropna()

    # print(f"Meta-model {name} data: {df.shape}")
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

    if 'horizon' not in df.columns:
        df['horizon'] = \
            (df.index.hour*2 + df.index.minute/30).round().astype(np.int16)
            # (df.index.hour + df.index.minute/60) / 24

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
    # meta_net,
    df_train,
    df_valid,
    feature_cols,
    # pred_length : int,
    valid_length: int,
    dropout     : float,
    num_cells   : int,
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

    # Initialize meta-model
    context_cols = [c for c in feature_cols
                    if c not in ['consumption_nn', 'consumption_lr', 'consumption_rf']]

    meta_nets  = []
    optimizers = []
    schedulers = []

    for h in range(48):
        net = MetaNet(context_dim=len(context_cols)+1, # including `horizon`
                      num_predictors=3,
                     dropout=dropout, num_cells=num_cells).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', patience=patience, factor=factor)  #, verbose=True

        meta_nets .append(net)
        optimizers.append(opt)
        schedulers.append(sch)


    criterion = nn.MSELoss()

    best_valid_loss = float('inf')
    _feature_cols = feature_cols + ['horizon']




    for epoch in range(epochs):
        epoch_weights    = []
        train_loss_total = 0.;   len_train_dataset = 0.
        valid_loss_total = 0.;   len_valid_dataset = 0.

        for h in range(valid_length):
            df_train_h = df_train[df_train['horizon'] == h].drop(columns=['horizon'])
            valid_loss_h = 0.

            preds_train, context_train, y_train = to_tensors(df_train_h, _feature_cols)
            train_dataset = TensorDataset(preds_train, context_train, y_train)

            train_loader = DataLoader(
                train_dataset,
                batch_size= batch_size,
                shuffle   = True,
                drop_last = True
            )

            net = meta_nets [h]
            opt = optimizers[h]
            sch = schedulers[h]


            # Training
            net.train()

            for preds_b, context_b, y_b in train_loader:

                opt.zero_grad()

                # Forward pass
                y_meta, weights = net(context_b.to(device),
                                        preds_b.to(device))
                # Loss
                loss_train = criterion(y_meta, y_b.to(device))

                # Backward
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                opt.step()

                train_loss_total  += loss_train.item() * len(y_b)
                len_train_dataset += len(y_b)


            # Validation
            df_valid_h = df_valid[df_valid['horizon'] == h]
            preds_valid, context_valid, y_valid = to_tensors(df_valid_h, _feature_cols)
            valid_dataset= TensorDataset(preds_valid, context_valid, y_valid)
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=batch_size*2, shuffle=False)

            net.eval()

            with torch.no_grad():
                for preds_b, context_b, y_b in valid_loader:
                    y_meta_b, weights = net(context_b.to(device),
                                              preds_b.to(device))
                    epoch_weights.append(weights.cpu())
                    loss_valid = criterion(y_meta_b, y_b.to(device))

                    valid_loss_total  += loss_valid.item() * len(y_b)  # whole epoch
                    valid_loss_h      += loss_valid.item() * len(y_b)  # this h
                    len_valid_dataset += len(y_b)


            # Learning rate scheduling
            sch.step(valid_loss_h)


        train_loss_avg = train_loss_total / len_train_dataset
        valid_loss_avg = valid_loss_total / len_valid_dataset
        # Logging


        if (epoch + 1) % 2 == 0 or epoch == 0:
            all_w = torch.cat(epoch_weights, dim=0)
            avg_weights = all_w.mean(dim=0).numpy()
            print(f"Epoch{epoch+1:3n}/{epochs}: "
                  f"losses train{train_loss_avg:6.2f}, "
                  f"valid{valid_loss_avg:6.2f}; "
                  f"avg weights: NN{avg_weights[0]*100:5.1f}%, "
                  f"LR{avg_weights[1]*100:5.1f}%, RF{avg_weights[2]*100:5.1f}%")

        # Save best model
        if valid_loss_avg < best_valid_loss:
            best_valid_loss = valid_loss_avg
            # torch.save(net.state_dict(), 'cache/best_metamodel.pth')

    # Load best model
    # load_state_dict(torch.load('cache/best_metamodel.pth', weights_only=True))

    return meta_nets




def metamodel_NN(list_dict_pred, list_dict_baseline, list_X, list_y, list_dates,
                feature_cols,
                #constants
                # pred_length : int,
                valid_length: int,
                dropout     : float,
                num_cells   : int,
                epochs      : int,
                lr          : float,
                weight_decay: float,
                patience    : int,
                factor      : float,
                batch_size  : int,
                device):

    [dict_pred_train_GW, dict_pred_valid_GW, dict_pred_test_GW] = list_dict_pred
    [dict_baseline_train_GW,dict_baseline_valid_GW,dict_baseline_test_GW] = \
        list_dict_baseline
    [X_train_GW,  X_valid_GW, X_test_GW] = list_X
    [y_train_GW,  y_valid_GW, y_test_GW] = list_y
    [train_dates, valid_dates,test_dates]= list_dates


    _feature_cols = feature_cols + ['horizon']

    # Prepare data
    df_meta_train = prepare_meta_data(
        "train",
        dict_pred_train_GW,
        dict_baseline_train_GW,
        X_train_GW,
        y_train_GW,
        train_dates,
        feature_cols  # /!\ 'horizon' will be added inside
    )

    df_meta_valid = prepare_meta_data(
        "valid",
        dict_pred_valid_GW,
        dict_baseline_valid_GW,
        X_valid_GW,
        y_valid_GW,
        valid_dates,
        feature_cols  # /!\ 'horizon' will be added inside
    )



    # meta_net = MetaNet(context_dim=len(context_cols), num_predictors=3,
    #                              dropout=dropout, num_cells=num_cells)

    # Train
    meta_nets = train_meta_model(
        # meta_net,
        df_meta_train, df_meta_valid,
        feature_cols,
        valid_length,
        dropout,
        num_cells,
        epochs,
        lr, weight_decay,
        batch_size,
        patience, factor,
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
        feature_cols  # /!\ 'horizon' will be added inside
    )

    preds_train, context_train, y_train = to_tensors(df_meta_train, _feature_cols)
    preds_valid, context_valid, y_valid = to_tensors(df_meta_valid, _feature_cols)
    preds_test,  context_test,  y_test =  to_tensors(df_meta_test,  _feature_cols)



    pred_meta2_train = torch.zeros(len(df_meta_train), device=device)
    pred_meta2_valid = torch.zeros(len(df_meta_valid), device=device)
    pred_meta2_test  = torch.zeros(len(df_meta_test ), device=device)


    weights_train_all = []
    weights_valid_all = []
    weights_test_all  = []

    with torch.no_grad():
        for h in range(valid_length):
            net = meta_nets[h]
            net.eval()

            # ---- TRAIN ----
            idx = (df_meta_train['horizon'].values == h)
            if idx.any():
                y_hat, w = net(
                    context_train[idx].to(device),
                    preds_train[idx].to(device)
                )
                pred_meta2_train[idx] = y_hat
                weights_train_all.append(w.cpu())

            # ---- VALID ----
            idx = (df_meta_valid['horizon'].values == h)
            if idx.any():
                y_hat, w = net(
                    context_valid[idx].to(device),
                    preds_valid[idx].to(device)
                )
                pred_meta2_valid[idx] = y_hat
                weights_valid_all.append(w.cpu())

            # ---- TEST ----
            idx = (df_meta_test['horizon'].values == h)
            if idx.any():
                y_hat, w = net(
                    context_test[idx].to(device),
                    preds_test[idx].to(device)
                )
                pred_meta2_test[idx] = y_hat
                weights_test_all.append(w.cpu())


    # Evaluate
    rmse_test = torch.sqrt(torch.mean((pred_meta2_test - y_test.to(device))**2))
    print(f"\nTest RMSE: {rmse_test.item():.2f} GW")

    # Analyze learned weights
    weights_test_all = torch.cat(weights_test_all, dim=0)  # (N_total, 3)
    avg_weights_test = weights_test_all.mean(dim=0)
    print(f"Average test weights: NN={avg_weights_test[0]*100:.1f}%, "
          f"LR={avg_weights_test[1]*100:.1f}%, RF={avg_weights_test[2]*100:.1f}%")


    return (pd.Series(pred_meta2_train.cpu().numpy(), index=df_meta_train.index),
            pd.Series(pred_meta2_valid.cpu().numpy(), index=df_meta_valid.index),
            pd.Series(pred_meta2_test .cpu().numpy(), index=df_meta_test .index))

