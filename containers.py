import time

from   dataclasses import dataclass, field, InitVar

from   typing      import Dict, List, Tuple, Optional, Any

import torch
from   torch       import nn  #, optim
# from   torch.optim.lr_scheduler import _LRScheduler

import numpy  as np
import pandas as pd


import architecture, utils, metamodel, plots


@dataclass
class DataSplit:
    name:             str             # 'train'   | 'valid'     | 'test'
    name_display:     str             # 'training'| 'validation'| 'testing'

    # Indices into the original series
    idx:              pd.Index | range

    # Core data
    X:                np.ndarray  # torch.Tensor
    y_nation:         np.ndarray
    Y_regions:        np.ndarray

    dates:            pd.DatetimeIndex
    Tavg_degC:        np.ndarray

    true_nation_GW:   Optional[np.ndarray]
    X_columns:        Optional[List[str]]     = None

    # Scaled versions
    # X_scaled_dev:     Optional[torch.Tensor]  = None
    # y_nation_scaled_dev:Optional[torch.Tensor]= None
    # Y_regions:    Optional[torch.Tensor]  = None

    # Torch plumbing
    dataset_scaled:   Optional[torch.utils.data.Dataset   ] = None
    loader:           Optional[torch.utils.data.DataLoader] = None

    # input to metamodels
    input_metamodel_LR:Optional[Dict[str, pd.DataFrame]] = None
    y_metamodel_LR:    Optional[Dict[str, pd.Series   ]] = None

    # predictions (column: name/type of model) -- added later
    dict_preds_ML:   Optional[Dict[str, pd.Series]] = field(default_factory=dict)
    dict_preds_NNTQ: Optional[Dict[str, pd.Series]] = field(default_factory=dict)
    dict_preds_meta: Optional[Dict[str, pd.Series]] = field(default_factory=dict)


    def __len__(self) -> int:
        return len(self.y_nation)


    @property
    def start(self):
        return self.dates[0]

    @property
    def end(self):
        return self.dates[-1]


    def prediction_day_ahead(self, model, scaler_y_nation,
            cols_features: List[str], device,
            input_length: int, pred_length: int, valid_length: int,
            minutes_per_step: int, quantiles: Tuple[float, ...]) -> None:

        # print(self.name)
        if self.name == 'complete':
            self.true_nation_GW = pd.Series(self.y_nation, index=self.dates)
            return

        # train, valid, test: predict day ahead
        self.true_nation_GW, self.dict_preds_NNTQ, self.dict_preds_ML, self.origin_times = \
            utils.subset_predictions_day_ahead(self.X, self.loader,
                model, scaler_y_nation, cols_features, device,
                input_length, pred_length, valid_length, minutes_per_step, quantiles)

    # print(pd.concat([self.true_nation_GW, pd.Series(self.y_nation,index=self.dates)],axis=1))


    def worst_days_by_loss(self,
                           temperature_full: pd.Series,
                           holidays_full   : pd.Series,
                           num_steps_per_day:int,
                           top_n           : int,
                           verbose         : int = 0) -> (pd.DataFrame, float):
        return utils.worst_days_by_loss(
            split       = self.name,
            y_true      = self.true_nation_GW,
            y_pred      = self.dict_preds_NNTQ['q50'],
            temperature = temperature_full.iloc[self.idx],
            holidays    = holidays_full   .iloc[self.idx],
            num_steps_per_day=num_steps_per_day,
            top_n       = top_n,
            verbose     = verbose
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
    def calculate_input_metamodel_LR(self, split_active: str) -> None:
        if self.name == 'complete':
            self.input_metamodel_LR= None
            self.y_metamodel_LR    = None
            return

        # train, valide, test: prepare input
        _y_nation_true = pd.Series(self.y_nation.squeeze(),
                                   name='y_nation', index=self.dates)

        _input = pd.concat([_y_nation_true,
                            pd.DataFrame(self.dict_preds_ML),
                            self.dict_preds_NNTQ['q50']
                           ],  axis=1, join="inner").astype('float32')
        _input.columns = ['y_nation'] + list(self.dict_preds_ML.keys()) + ['NNTQ']
        self.input_metamodel_LR = _input.drop(columns=['y_nation'])

        if self.name == split_active:
            self.y_metamodel_LR = _input[['y_nation']].squeeze()  #.to_numpy()

    # compare models
    def compare_models(self, unit: str = "GW", verbose: int = 0) -> pd.DataFrame:
        if verbose > 0:
            print(f"\n{self.name_display:10s} metrics [{unit}]:")
        return utils.compare_models( self.true_nation_GW,self.dict_preds_NNTQ,
                                     self.dict_preds_ML, self.dict_preds_meta,
                                     subset=self.name, unit=unit, verbose=verbose)


    # plotting
    def plots_diagnostics(self,
                          names_baseline:    str,
                          names_meta:        str,
                          temperature_full:  pd.Series,
                          num_steps_per_day: int,
                          quantiles:         List[str]):
        # print(self.name)
        # print(self.true_nation_GW.shape, self.true_nation_GW)
        # print(self.dict_preds_NNTQ ['q50'].shape, self.dict_preds_NNTQ ['q50'])
        # print(self.dict_preds_meta['LR'].shape, self.dict_preds_meta['LR'])

        plots.diagnostics(self.name,
            self.true_nation_GW, {q: self.dict_preds_NNTQ[q] for q in quantiles},
            self.dict_preds_ML, self.dict_preds_meta,
            names_baseline, names_meta, temperature_full.iloc[self.idx],
            num_steps_per_day)



@dataclass
class DatasetBundle:
    train:   DataSplit
    valid:   DataSplit
    test:    DataSplit
    complete:DataSplit

    # whole set, unscaled
    X:        np.ndarray
    y_nation: np.ndarray
    Y_regions:np.ndarray

    # size of the input data
    num_features  : int
    weights_regions: Dict[str, float]
    num_time_steps: int

    # time increment
    minutes_per_step:  int
    num_steps_per_day: int

    scaler_y_nation:   object
    scaler_X: Optional[object] = None

    # metamodels (added later)
    weights_meta_LR:     Optional[str]        = None
    split_active_meta_LR:Optional[List[float]]= None
    weights_meta_NN:     Optional[str]        = None


    def items(self):
        return {
            'train':   self.train,
            'valid':   self.valid,
            'test':    self.test,
            'complete':self.complete
        }.items()

    def predictions_day_ahead(self, model, scaler_y,
            cols_features: List[str], device,
            input_length: int, pred_length: int, valid_length: int,
            minutes_per_step: int, quantiles: Tuple[float, ...]) -> None:
        # print("predictions_day_ahead")
        for (_name_split, _data_split) in self.items():
            # print(_name_split)
            _data_split.prediction_day_ahead(
                model, scaler_y, cols_features, device,
                input_length, pred_length, valid_length, minutes_per_step, quantiles)

    # Metamodels
    def calculate_metamodel_LR(self,
                               split_active:str,
                               min_weight:  float = 0.,
                               verbose:     int   = 0) -> None:
        assert split_active in ['train', 'valid'], split_active
        self.weights_meta_LR = split_active

        for (_split, _data_split) in self.items():
            _data_split.calculate_input_metamodel_LR(split_active=split_active)
            # print(_split, _data_split.input_metamodel_LR is not None,
            #               _data_split.    y_metamodel_LR is not None)

        if split_active == 'train':
            (self.weights_meta_LR,
             self.train.dict_preds_meta['LR'],
             self.valid.dict_preds_meta['LR'],
             self.test .dict_preds_meta['LR'])= metamodel.weights_LR_metamodel(
                    self.train.input_metamodel_LR, self.train.    y_metamodel_LR,
                    self.valid.input_metamodel_LR, self.test .input_metamodel_LR,
                    min_weight=min_weight, verbose=verbose)
        else:  # on valid
            (self.weights_meta_LR,
             self.valid.dict_preds_meta['LR'],
             self.test .dict_preds_meta['LR'], _) =\
                metamodel.weights_LR_metamodel(
                    self.valid.input_metamodel_LR, self.valid.y_metamodel_LR,
                    self.test .input_metamodel_LR, None,
                    min_weight=min_weight, verbose=verbose)


    def calculate_metamodel_NN(self,
                    cols_features: List[str],
                    valid_length: int,
                    split_active: str,
                    metamodel_nn_parameters: Dict[str, Any],
                    verbose     : int = 0) -> None:
        assert split_active in ['train', 'valid'], split_active

        if split_active == 'train':
            (self.train.dict_preds_meta['NN'],
             self.valid.dict_preds_meta['NN'],
             self.test .dict_preds_meta['NN'],
             list_models, self.avg_weights_meta_NN) = \
                metamodel.metamodel_NN(
                    self.train, self.valid, self.test,
                    cols_features, valid_length, metamodel_nn_parameters, verbose)
        else:  # on valid
            # the second argumennt is actual validation (learning rate scheduling)
            #   /!\ use train for that? or nothing?
            (self.valid.dict_preds_meta['NN'], _,
             self.test .dict_preds_meta['NN'],
             list_models, self.avg_weights_meta_NN) = \
                metamodel.metamodel_NN(
                    self.valid, self.train, self.test,
                    cols_features, valid_length, metamodel_nn_parameters, verbose)




@dataclass
class NeuralNet:

    device           : object

    weights_regions  : Dict[str, float]

    batch_size       : int
    epochs           : int
    patience         : int

    # optimizer
    learning_rate    : float
    weight_decay     : float
    dropout          : float
    warmup_steps     : Optional[int]   = None

    # rest of early stopping
    min_delta        : Optional[float] = None
    factor           : Optional[float] = None    # metamodel only

    # architecture
    model_dim        : Optional[int]   = None
    num_layers       : Optional[int]   = None
    num_heads        : Optional[int]   = None
    ffn_size         : Optional[int]   = None
    num_cells        : Optional[List[int]]=None  # metamodel only

    # in steps of half-hours
    input_length     : Optional[int]   = None
    pred_length      : Optional[int]   = None
    valid_length     : Optional[int]   = None
    features_in_future:Optional[bool]  = None

    #
    patch_length     : Optional[int]   = None
    stride           : Optional[int]   = None
    num_patches      : Optional[int]   = None

    # geometric blocks
    num_geo_blocks   : Optional[int]   = None
    geo_block_ratio  : Optional[float] = None

    # quantile loss
    quantiles        : Optional[List[float]]= None
    lambda_cross     : Optional[float] = None
    lambda_coverage  : Optional[float] = None
    lambda_deriv     : Optional[float] = None
    lambda_median    : Optional[float] = None
    smoothing_cross  : Optional[float] = None
        # temperature-dependence (pinball loss, coverage penalty)
    saturation_cold_degC:Optional[float]=None
    threshold_cold_degC:Optional[float]= None
    lambda_cold      : Optional[float] = None

    lambda_regions   : Optional[float] = None



    # will be created in __post_init__
    model            : Optional[nn.Module]            = \
                         field(default=None, init=False, repr=False, compare=False)
    optimizer        : Optional[torch.optim.Optimizer]= \
                         field(default=None, init=False, repr=False, compare=False)
    scheduler        : Optional[torch.optim.lr_scheduler._LRScheduler] = \
                         field(default=None, init=False, repr=False, compare=False)
    early_stopping   : Optional[Dict[str, Any]]       = \
                         field(default_factory=dict,     repr=False, compare=False)
    amp_scaler       : Optional[torch.amp.GradScaler] = \
                         field(default=None, init=False, repr=False, compare=False)

    num_quantiles    : int         = field(default_factory=int)
    num_regions      : int         = field(default_factory=int)

    # needed by __post_init__
    len_train_data   : InitVar[Optional[int]]   = None
    num_features     : InitVar[Optional[int]]   = None



    def __post_init__(self,
                      len_train_data: Optional[int] = None,
                      num_features  : Optional[int] = None):
        if self.quantiles is None : return   # metamodel

        self.num_quantiles = len(self.quantiles) \
            if self.quantiles is not None else None

        self.num_regions   = len(self.weights_regions) \
            if self.weights_regions is not None else None

        self.model = architecture.TimeSeriesTransformer(
                num_features, self.model_dim, self.num_heads, self.num_layers,
                self.input_length, self.patch_length, self.stride, self.pred_length,
                self.features_in_future,
                self.dropout, self.ffn_size, self.num_quantiles, self.num_regions,
                self.num_geo_blocks, self.geo_block_ratio
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
                            self.model.parameters(),
                            lr          = self.learning_rate,
                            weight_decay= self.weight_decay)

        def my_lr_warmup_cosine(step):
            return architecture.lr_warmup_cosine(
                step, self.warmup_steps, self.epochs, len_train_data)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=my_lr_warmup_cosine
        )
        # print("Initial LR:", scheduler.get_last_lr())


        # Example usage in your training loop:
        # Initialize early stopping
        self.early_stopping = architecture.EarlyStopping(
            patience=self.patience, min_delta=self.min_delta)

        self.amp_scaler     = torch.amp.GradScaler(device=self.device)


    def training_loop(self,
                      train_loader,
                      valid_loader,
                      validate_every: int,
                      display_every : int,
                      plot_conv_every:int,
                      verbose       : int = 0):

        num_epochs: int = self.epochs

        t_epoch_loop_start = time.perf_counter()

        list_of_min_losses= (9.999, 9.999, 9.999)
        list_of_lists     = ([], [], [], [])

        # first_step = True

        for epoch in range(num_epochs):

            # Training
            t_train_start = time.perf_counter()

            train_loss_quantile_h_scaled, dict_train_loss_quantile_h = \
                architecture.subset_evolution_torch(
                    self, train_loader)  #, data.train.Tavg_degC)

            train_loss_quantile_h_scaled= \
                train_loss_quantile_h_scaled.detach().cpu().numpy()
            dict_train_loss_quantile_h  = {k: v.detach().cpu().numpy()
                               for k, v in dict_train_loss_quantile_h.items()}


            if verbose >= 2:
                print(f"training took:   {time.perf_counter() - t_train_start:.2f} s")


            # validation
            if ((epoch+1) % validate_every == 0) | (epoch == 0):

                t_valid_start     = time.perf_counter()
                valid_loss_quantile_h_scaled, dict_valid_loss_quantile_h = \
                    architecture.subset_evolution_numpy(
                        self, valid_loader)  #, data.valid.Tavg_degC)

                if verbose >= 2:
                    print(f"validation took: {time.perf_counter()-t_valid_start:.2f} s")


            # display evolution of losses
            (list_of_min_losses, list_of_lists) = \
                utils.display_evolution(
                    epoch, t_epoch_loop_start,
                    train_loss_quantile_h_scaled.mean(),
                    valid_loss_quantile_h_scaled.mean(),
                    list_of_min_losses, list_of_lists,
                    num_epochs, display_every, plot_conv_every,
                    self.min_delta, verbose)

            # plotting convergence
            if ((epoch+1 == plot_conv_every) | ((epoch+1) % plot_conv_every == 0))\
                    & (epoch < num_epochs-2) & verbose > 0:
                plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                                  list_of_lists[2], list_of_lists[3],
                                  partial=True, verbose=verbose)

            # Check for early stopping
            if self.early_stopping(valid_loss_quantile_h_scaled.mean()):
                if verbose > 0:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        # torch.cuda.empty_cache()

        return (list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
                dict_valid_loss_quantile_h)


    def run(
            self,
            data             : DatasetBundle,
            cols_features     : List[str],
            temperature_full : pd.Series,  # TODO np?
            holidays_full    : pd.Series,
            minutes_per_step : int,
            validate_every   : int,
            display_every    : int,
            plot_conv_every  : int,
            cache_dir        : str,
            num_worst_days   : int,
            verbose          : int = 0
        ):

        if verbose > 0:
            if cache_dir is None:
                print("Training NNTQ (calculation forced)...")
            else:
                print("Training NNTQ (no cache found)...")


        # loop for training and validation
        if verbose > 0:
            print("Starting training...")

        list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
            dict_valid_loss_quantile_h = self.training_loop(
                data.train.loader, data.valid.loader,
                validate_every, display_every, plot_conv_every, verbose)


        # plotting convergence for entire training
        if verbose > 0:
            plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                                       list_of_lists[2], list_of_lists[3],
                                       partial=False, verbose=verbose)

            plots.loss_per_horizon(dict({"total": valid_loss_quantile_h_scaled}, \
                                   **dict_valid_loss_quantile_h), minutes_per_step,
                                   "validation loss")


        # test loss
        test_loss_quantile_h_scaled, dict_test_loss_quantile_h = \
           architecture.subset_evolution_numpy(self, data.test.loader)

        if verbose >= 3:
            print(pd.DataFrame(dict({"total": test_loss_quantile_h_scaled}, \
                                    **dict_test_loss_quantile_h)
                               ).round(2).to_string())
        if verbose > 0:
            plots.loss_per_horizon(dict({"total": test_loss_quantile_h_scaled}, \
                                         **dict_test_loss_quantile_h), minutes_per_step,
                                   "test loss")


        t_metamodel_start = time.perf_counter()

        data.predictions_day_ahead(
                self.model, data.scaler_y_nation,
                cols_features = cols_features,
                device       = self.device,
                input_length = self.input_length,
                pred_length  = self.pred_length,
                valid_length = self.valid_length,
                minutes_per_step=minutes_per_step,
                quantiles    = self.quantiles
            )

        num_steps_per_day = int(round(24*60/minutes_per_step))
        worst_days_test_df, avg_abs_worst_days_test_NN_median = \
            data.test.worst_days_by_loss(
                temperature_full = temperature_full,
                holidays_full    = holidays_full,
                num_steps_per_day= num_steps_per_day,
                top_n            = num_worst_days,
                verbose          = verbose
            )
        if verbose >= 3:
            print(worst_days_test_df.to_string())



        # ============================================================
        # TEST PREDICTIONS
        # ============================================================

        if verbose > 0:
            print("\nStarting test ...")  #"(baseline: {name_baseline})...")

            print("\nTesting quantiles")

        quantile_delta_coverage = {}
        for tau in self.quantiles:
            key = f"q{int(100*tau)}"
            cov = utils.quantile_coverage(data.test.true_nation_GW,
                                          data.test.dict_preds_NNTQ[key])
            quantile_delta_coverage[key] = cov-tau

            if verbose > 0:
                print(f"Coverage {key}:{cov*100:5.1f}%, i.e."
                      f"{(cov-tau)*100:5.1f}%pt off{tau*100:3n}% target")

        if verbose > 0:
            print()


        return (data, quantile_delta_coverage, avg_abs_worst_days_test_NN_median)
