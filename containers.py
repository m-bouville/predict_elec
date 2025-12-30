from   dataclasses import dataclass, field

from   typing      import Dict, List, Optional

import torch

import numpy  as np
import pandas as pd




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