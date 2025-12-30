from   dataclasses import dataclass, field

from   typing      import Dict, List, Tuple, Optional

import torch

import numpy  as np
import pandas as pd


import utils, metamodel, plots


@dataclass
class DataSplit:
    name: str                      # 'train' | 'valid' | 'test'

    # Indices into the original series
    idx:            pd.Index | range

    # Core data
    X_dev:          torch.Tensor
    y_dev:          torch.Tensor
    dates:          pd.DatetimeIndex

    # Scaled versions
    X_scaled_dev:   Optional[torch.Tensor] = None
    y_scaled_dev:   Optional[torch.Tensor] = None

    # Torch plumbing
    dataset_scaled: Optional[torch.utils.data.Dataset   ] = None
    loader:         Optional[torch.utils.data.DataLoader] = None

    # predictions (colum: name/type of model) -- added later
    dict_preds_ML:  Optional[Dict[str, pd.Series]] = field(default_factory=dict)
    dict_preds_NN:  Optional[Dict[str, pd.Series]] = field(default_factory=dict)
    dict_preds_meta:Optional[Dict[str, pd.Series]] = field(default_factory=dict)


    def __len__(self) -> int:
        return len(self.y_dev)


    @property
    def start(self):
        return self.dates[0]

    @property
    def end(self):
        return self.dates[-1]


    def prediction_day_ahead(self, model, scaler_y,
            feature_cols: List[str], device,
            input_length: int, pred_length: int, valid_length: int,
            minutes_per_step: int, quantiles: Tuple[float, ...]) -> None:
        self.true_GW, self.dict_preds_NN, self.dict_preds_ML, self.origin_times = \
            utils.subset_predictions_day_ahead(self.X_dev, self.loader,
                model, scaler_y, feature_cols, device,
                input_length, pred_length, valid_length, minutes_per_step, quantiles)

    def worst_days_by_loss(self,
                           temperature_full: pd.Series,
                           holidays_full   : pd.Series,
                           num_steps_per_day:int,
                           top_n           : int) -> pd.DataFrame:
        return utils.worst_days_by_loss(
            y_true      = self.true_GW,
            y_pred      = self.dict_preds_NN['q50'],
            temperature = temperature_full.iloc[self.idx],
            holidays    = holidays_full   .iloc[self.idx],
            num_steps_per_day=num_steps_per_day,
            top_n       = top_n,
        )

        # import matplotlib.pyplot as plt
        # day = top_bad_days.iloc[0]["date"]

        # q50_idx = QUANTILES.index(0.5)

        # day = top_bad_days.iloc[0]["date"]
        # mask_day = dates_masked.normalize() == day

        # # inverse-scale for plotting
        # y_true_day_GW = scaler_y.inverse_transform(
        #     y_true_masked.reshape(-1, 1)
        # ).ravel()

        # y_pred_day_GW = scaler_y.inverse_transform(
        #     y_pred_masked[:, q50_idx].reshape(-1, 1)
        # ).ravel()

        # plt.figure(figsize=(10, 4))
        # plt.plot(dates_masked, y_true_day_GW, label="true")
        # plt.plot(dates_masked, y_pred_day_GW, label="pred (q50)")
        # plt.legend()
        # plt.title(f"Worst training day: {day}")
        # plt.show()



    # Metamodel LR
    def calculate_input_metamodel_LR(self) -> None:
        _y_true = pd.Series(self.y_dev, name='y', index=self.dates)

        _input = pd.concat([_y_true,
                            pd.DataFrame(self.dict_preds_ML),
                            self.dict_preds_NN['q50']
                           ],  axis=1, join="inner").astype('float32')
        _input.columns = ['y'] + list(self.dict_preds_ML.keys()) + ['nn']
        self.input_metamodel_LR = _input.drop(columns=['y'])

        if self.name == 'train':
            self.y_metamodel_LR = _input[['y']].squeeze()  #.to_numpy()

    # compare models
    def compare_models(self, unit: str = "GW", verbose: int = 0):
        utils.compare_models(self.true_GW,      self.dict_preds_NN,
                             self.dict_preds_ML,self.dict_preds_meta,
                             subset=self.name, unit=unit, verbose=verbose)


    # plotting
    def plots_diagnostics(self,
                          name_baseline:     str,
                          temperature_full:  pd.Series,
                          num_steps_per_day: int):
        plots.diagnostics(self.true_GW, {'q50': self.dict_preds_NN['q50']},
                self.dict_preds_ML, self.dict_preds_meta['meta_NN'],
                name_baseline, temperature_full.iloc[self.idx],
                num_steps_per_day)



@dataclass
class DatasetBundle:
    train:  DataSplit
    valid:  DataSplit
    test:   DataSplit

    # whole set, unscaled
    X:      np.ndarray
    y:      np.ndarray

    scaler_y: object
    scaler_X: Optional[object] = None

    # added later
    weights_meta_LR: Optional[List[float]] = None


    def items(self):
        return {
            'train': self.train,
            'valid': self.valid,
            'test':  self.test
        }.items()


    def predictions_day_ahead(self, model, scaler_y,
            feature_cols: List[str], device,
            input_length: int, pred_length: int, valid_length: int,
            minutes_per_step: int, quantiles: Tuple[float, ...]) -> None:
        for (_split, _data_split) in self.items():
            _data_split.prediction_day_ahead(
                model, scaler_y, feature_cols, device,
                input_length, pred_length, valid_length, minutes_per_step, quantiles)

    # Metamodels
    def calculate_inputs_metamodel_LR(self) -> None:
        for (_split, _data_split) in self.items():
            _data_split.calculate_input_metamodel_LR()

    def calculate_metamodel_LR(self, verbose: int=0) -> None:
        self.calculate_inputs_metamodel_LR()

        (self.weights_meta_LR,
        self.train.dict_preds_meta['meta_LR'],
        self.valid.dict_preds_meta['meta_LR'],
        self.test .dict_preds_meta['meta_LR']) = \
            metamodel.weights_LR_metamodel(self.train.input_metamodel_LR,
                                           self.train.    y_metamodel_LR,
                                           self.valid.input_metamodel_LR,
                                           self.test .input_metamodel_LR,
                                           verbose=verbose)

    def calculate_metamodel_NN(self,
                    feature_cols: List[str],
                    valid_length: int,
                    dropout     : float,
                    num_cells   : int,
                    epochs      : int,
                    lr          : float,
                    weight_decay: float,
                    patience    : int,
                    factor      : float,
                    batch_size  : int,
                    device) -> None:
        (self.train.dict_preds_meta['meta_NN'],
         self.valid.dict_preds_meta['meta_NN'],
         self.test .dict_preds_meta['meta_NN']) = \
            metamodel.metamodel_NN(
                self.train, self.valid, self.test,
                feature_cols, valid_length,
                #constants
                dropout, num_cells, epochs,
                lr, weight_decay,
                patience, factor,
                batch_size, device)


