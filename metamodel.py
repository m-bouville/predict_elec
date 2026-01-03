from   typing import Dict, Tuple, List, Optional, Any


import torch
import torch.nn as nn
from   torch.utils.data     import TensorDataset, DataLoader

import numpy  as np
import pandas as pd

from   sklearn.linear_model import LinearRegression
from   scipy.optimize       import minimize


import plots # losses  # architecture



# ----------------------------------------------------------------------
# linear regression for metamodel
# ----------------------------------------------------------------------


# custom LR enforcing a minimum weight
class ConstrainedLinearRegression(LinearRegression):
    def __init__(self, min_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_weight = min_weight

    def fit(self, X, y):
        n_samples, n_features = X.shape

        def objective(coef):
            return np.sum((y - X.dot(coef))**2)

        initial_coef = np.zeros(n_features)
        bounds = [(self.min_weight, None) for _ in range(n_features)]

        result = minimize(objective, initial_coef, bounds=bounds, method='L-BFGS-B')
        self.coef_ = result.x
        self._set_intercept(X, y)
        return self

    def _set_intercept(self, X, y):
        self.intercept_ = np.mean(y - X.dot(self.coef_))


def weights_LR_metamodel(X_input  : pd.DataFrame,
                         y_input  : pd.Series,
                         X_pred1  : Optional[pd.DataFrame] = None,
                         X_pred2  : Optional[pd.DataFrame] = None,
                         min_weight:float= 0.,
                         verbose  : int  = 0) \
        -> Tuple[Dict[str, float], pd.Series, pd.Series | None, pd.Series | None]:
    # X: (N, 4), y: (N,)

    model_meta = ConstrainedLinearRegression(
        fit_intercept=False, min_weight=min_weight)
    model_meta.fit(X_input, y_input)

    if verbose >= 3:
        pred_input = model_meta.predict(X_input)
        _output = pd.concat([
             y_input, X_input,
             pd.Series(pred_input, name='meta', index=y_input.index)], axis=1)
        _output.columns=['true'] + X_input.columns + ['meta']
        print(_output.astype(np.float32).round(2))

    weights_meta = {name: round(float(coeff), 3) for name, coeff in \
           zip(X_input.columns, model_meta.coef_)}

    pred_input= pd.Series(model_meta.predict(X_input), index=X_input.index)
    pred1     = pd.Series(model_meta.predict(X_pred1), index=X_pred1.index) \
                    if X_pred1 is not None else None
    pred2     = pd.Series(model_meta.predict(X_pred2), index=X_pred2.index) \
                    if X_pred2 is not None else None

    # print(f"types: {type(pred_input)}, {type(pred_valid)}, {type(pred_test)}")

    return (weights_meta, pred_input, pred1, pred2)





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
            preds:   (B, 4) - predictions from [nn, lr, rf, gb]

        Returns:
            y_meta:  (B,)   - weighted combination
            weights: (B, 4) - learned weights
        """
        # Compute weights from context
        logits  = self.net(context)              # (B, 4)
        weights = torch.softmax(logits, dim=-1)  # (B, 4)

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
        'NN':    dict_pred_GW['q50'],
        'LR':    dict_baseline_GW.get('LR', np.nan),
        'RF':    dict_baseline_GW.get('RF', np.nan),
        'GB':    dict_baseline_GW.get('GB', np.nan),
    }, index=dates)

    # Add context features
    df_features = pd.DataFrame(X_features_GW, index=dates, columns=feature_cols)
    if 'horizon' not in df_features.columns:
        df_features['horizon']=(df_features.index.hour*2 + \
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
        df[['NN', 'LR', 'RF', 'GB']].values,
        dtype=torch.float32
    )

    # Context features (exclude predictions and target)

    if 'horizon' not in df.columns:
        df['horizon'] = \
            (df.index.hour*2 + df.index.minute/30).round().astype(np.int16)
            # (df.index.hour + df.index.minute/60) / 24

    context_cols = [c for c in feature_cols
        if c not in ['consumption_nn', 'consumption_lr',
                     'consumption_rf', 'consumption_gb']]
    context = torch.tensor(
        df[context_cols].values,
        dtype=torch.float32
    )

    # Target
    y_true = torch.tensor(df['y_true'].values, dtype=torch.float32)

    return preds, context, y_true


def train_meta_model(
    # meta_net,
    df_train    : pd.DataFrame,
    df_valid    : Optional[pd.DataFrame],
    feature_cols,
    valid_length: int,
    dropout     : float,
    num_cells   : int,
    epochs      : int,
    learning_rate:float,
    weight_decay: float,
    batch_size  : int,
    patience    : int,
    factor      : float,
    device,
    verbose     : int = 0
) -> Tuple[list, np.ndarray]:
    """
    Train meta-model on training data, validate on validation data.
    """

    # Initialize meta-model
    context_cols = [c for c in feature_cols
                    if c not in ['consumption_nn', 'consumption_lr',
                                 'consumption_rf', 'consumption_gb']]

    meta_nets  = []
    optimizers = []
    schedulers = []

    for h in range(48):
        net_h = MetaNet(context_dim=len(context_cols)+1, # including `horizon`
                        num_predictors=4, dropout=dropout,
                        num_cells=num_cells
                       ).to(device)
        opt_h = torch.optim.Adam(net_h.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
        sch_h = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt_h, mode='min', patience=patience, factor=factor)

        meta_nets .append(net_h)
        optimizers.append(opt_h)
        schedulers.append(sch_h)


    criterion = nn.MSELoss()

    best_train_loss = float('inf');   best_valid_loss = float('inf')
    _feature_cols = feature_cols + ['horizon']


    for epoch in range(epochs):
        epoch_weights    = []
        train_loss_total = 0.;   len_train_dataset = 0.
        valid_loss_total = 0.;   len_valid_dataset = 0.

        for h in range(valid_length):
            df_train_h = df_train[df_train['horizon']==h].drop(columns=['horizon'])
            train_loss_h = 0.;   valid_loss_h = 0.

            preds_train, context_train, y_train = \
                    to_tensors(df_train_h, _feature_cols)
            train_dataset = TensorDataset(preds_train, context_train, y_train)

            train_loader = DataLoader(
                train_dataset,
                batch_size= batch_size,
                shuffle   = True,
                drop_last = True
            )

            net_h = meta_nets [h]
            opt_h = optimizers[h]
            sch_h = schedulers[h]


            # Training
            net_h.train()

            for preds_b, context_b, y_b in train_loader:

                opt_h.zero_grad()

                # Forward pass
                y_meta, weights = net_h(context_b.to(device),
                                        preds_b.to(device))
                # Loss
                loss_train = criterion(y_meta, y_b.to(device))

                # Backward
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(net_h.parameters(), 1.)
                opt_h.step()

                train_loss_total += loss_train.item() * len(y_b)  # whole epoch
                train_loss_h     += loss_train.item() * len(y_b)  # this h
                len_train_dataset+= len(y_b)


            # Validation
            if df_valid is not None:
                df_valid_h = df_valid[df_valid['horizon'] == h]
                preds_valid, context_valid, y_valid = \
                        to_tensors(df_valid_h, _feature_cols)
                valid_dataset= TensorDataset(preds_valid, context_valid, y_valid)
                valid_loader = DataLoader(valid_dataset,
                                          batch_size=batch_size*2, shuffle=False)

                net_h.eval()

                with torch.no_grad():
                    for preds_b, context_b, y_b in valid_loader:
                        y_meta_b, weights = net_h(context_b.to(device),
                                                  preds_b.to(device))
                        epoch_weights.append(weights.cpu())
                        loss_valid = criterion(y_meta_b, y_b.to(device))

                        valid_loss_total+= loss_valid.item() * len(y_b) # whole epoch
                        valid_loss_h    += loss_valid.item() * len(y_b) # this h
                        len_valid_dataset+= len(y_b)

                # Learning rate scheduling
                sch_h.step(valid_loss_h)

            else:
                # Learning rate scheduling
                sch_h.step(train_loss_h)
        # end loop over h

        train_loss_avg = train_loss_total / len_train_dataset
        valid_loss_avg = valid_loss_total / len_valid_dataset \
                    if df_valid is not None else None

        # Logging
        if ((epoch + 1) % 2 == 0 or epoch == 0) and len(epoch_weights) > 0:
            all_w = torch.cat(epoch_weights, dim=0)
            avg_weights = all_w.mean(dim=0).numpy()
            if verbose > 0:
                print(f"Epoch{epoch+1:3n}/{epochs}: "
                      f"losses train{train_loss_avg:5.2f}, "
                      f"valid{valid_loss_avg:5.2f}; "
                      f"avg w: NN{avg_weights[0]*100:5.1f}%, "
                      f"LR{avg_weights[1]*100:5.1f}%, RF{avg_weights[2]*100:5.1f}%, "
                      f"GB{avg_weights[3]*100:5.1f}%")

        # Save best model
        if df_valid is not None:
            if valid_loss_avg < best_valid_loss:
                best_valid_loss = valid_loss_avg
                # torch.save(net.state_dict(), 'cache/best_metamodel.pth')
        elif train_loss_avg < best_train_loss:
                best_train_loss = train_loss_avg

    # Load best model
    # load_state_dict(torch.load('cache/best_metamodel.pth', weights_only=True))

    return meta_nets, all_w.numpy()



def metamodel_NN(data_train,
                 data_valid : Optional,
                 data_test  : Optional,
                 feature_cols: List[str],
                 valid_length: int,
                 metamodel_nn_parameters: Dict[str, Any],
                 verbose     : int = 0) \
        -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], list]:

    device = metamodel_nn_parameters['device']

    _feature_cols = feature_cols + ['horizon']

    # Prepare data
    df_meta_train = prepare_meta_data(
        "train",
        data_train.dict_preds_NN,
        data_train.dict_preds_ML,
        data_train.X_dev,
        data_train.y_dev,
        data_train.dates,
        feature_cols  # /!\ 'horizon' will be added inside
    )

    df_meta_valid = prepare_meta_data(
        "valid",
        data_valid.dict_preds_NN,
        data_valid.dict_preds_ML,
        data_valid.X_dev,
        data_valid.y_dev,
        data_valid.dates,
        feature_cols  # /!\ 'horizon' will be added inside
    ) if data_valid is not None else None



    # meta_net = MetaNet(context_dim=len(context_cols), num_predictors=3,
    #                              dropout=dropout, num_cells=num_cells)

    # Train
    meta_nets, weights = train_meta_model(
        # meta_net,
        df_meta_train, df_meta_valid,
        feature_cols,
        valid_length,
        dropout     = metamodel_nn_parameters['dropout'],
        num_cells   = metamodel_nn_parameters['num_cells'],
        epochs      = metamodel_nn_parameters['epochs'],
        learning_rate=metamodel_nn_parameters['learning_rate'],
        weight_decay= metamodel_nn_parameters['weight_decay'],
        batch_size  = metamodel_nn_parameters['batch_size'],
        patience    = metamodel_nn_parameters['patience'],
        factor      = metamodel_nn_parameters['factor'],
        device=device, verbose=verbose
    )

    # Test
    df_meta_test = prepare_meta_data(
        "test",
        data_test.dict_preds_NN,
        data_test.dict_preds_ML,
        data_test.X_dev,
        data_test.y_dev,
        data_test.dates,
        feature_cols  # /!\ 'horizon' will be added inside
    ) if data_test is not None else None

    preds_train, context_train, y_train = to_tensors(df_meta_train, _feature_cols)
    pred_meta_train = torch.zeros(len(df_meta_train), device=device)

    if data_valid is not None:
        preds_valid, context_valid, y_valid= to_tensors(df_meta_valid, _feature_cols)
        pred_meta_valid = torch.zeros(len(df_meta_valid), device=device)

    if data_test is not None:
        preds_test,  context_test,  y_test = to_tensors(df_meta_test,  _feature_cols)
        pred_meta_test  = torch.zeros(len(df_meta_test ), device=device)



    weights_train_h = []
    weights_valid_h = []
    weights_test_h  = []

    with torch.no_grad():
        for h in range(valid_length):
            net = meta_nets[h]
            net.eval()

            # ---- TRAIN ----
            idx = (df_meta_train['horizon'].values == h)
            if idx.any():
                y_hat, w = net(
                    context_train[idx].to(device),
                    preds_train  [idx].to(device)
                )
                pred_meta_train[idx] = y_hat
                weights_train_h.append(w.cpu().numpy().mean(axis=0)) # (n_models,)

            # ---- VALID ----
            if data_valid is not None:
                idx = (df_meta_valid['horizon'].values == h)
                if idx.any():
                    y_hat, w = net(
                        context_valid[idx].to(device),
                        preds_valid  [idx].to(device)
                    )
                    pred_meta_valid[idx] = y_hat
                    weights_valid_h.append(w.cpu().numpy().mean(axis=0))#(n_models,)

            # ---- TEST ----
            if data_test is not None:
                idx = (df_meta_test['horizon'].values == h)
                if idx.any():
                    y_hat, w = net(
                        context_test[idx].to(device),
                        preds_test  [idx].to(device)
                    )
                    pred_meta_test[idx] = y_hat
                    weights_test_h.append(w.cpu().numpy().mean(axis=0))#(n_models,)


    # Evaluate
    if data_test is not None and verbose >= 1:
        rmse_test = torch.sqrt(torch.mean((pred_meta_test - y_test.to(device))**2))
        print(f"\nTest RMSE: {rmse_test.item():.2f} GW")

        # Analyze learned weights
        df_weights_test  = pd.DataFrame(
                weights_test_h, columns=["NN", "LR", "RF", "GB"])

        plots.data(df_weights_test * 100, xlabel="horizon",
                   ylabel="weights NN metamodel [%}")
        avg_weights_test = df_weights_test.mean(axis=0)
            # (N_total, 4) ->
        print(f"Average test weights: NN={avg_weights_test['NN']*100:.1f}%,"
              f"LR={avg_weights_test['LR']*100:5.1f}%, "
              f"RF={avg_weights_test['RF']*100:5.1f}%, "
              f"GB={avg_weights_test['GB']*100:5.1f}%")



    return (pd.Series(pred_meta_train.cpu().numpy(), index=df_meta_train.index),
            pd.Series(pred_meta_valid.cpu().numpy(), index=df_meta_valid.index) \
                if data_valid is not None else None,
            pd.Series(pred_meta_test .cpu().numpy(), index=df_meta_test .index) \
                if data_test  is not None else None,
            meta_nets
            )

