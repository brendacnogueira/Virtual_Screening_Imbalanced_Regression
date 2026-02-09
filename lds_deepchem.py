from __future__ import annotations
import numpy as np
import deepchem as dc
from typing import Optional, Tuple
from lds_utils import LDSParams, make_lds_weighter_from_reference

def apply_lds_to_dc_splits(
    train: dc.data.Dataset,
    valid: Optional[dc.data.Dataset] = None,
    test: Optional[dc.data.Dataset] = None,
    lds: Optional[LDSParams] = None,
) -> Tuple[dc.data.Dataset, Optional[dc.data.Dataset], Optional[dc.data.Dataset]]:
    """
    Compute LDS weights from the TRAIN labels and attach them to the `.w` field of
    train/valid/test DeepChem datasets.

    Why this design?
    - We estimate the effective label density \tilde{p}(y) on the TRAIN labels only
      (to avoid leakage).
    - We then build a frozen "weighter" function that maps any y -> 1 / \tilde{p}(y),
      normalized to mean 1.0.
    - We apply that function to each split's labels and return new NumpyDatasets
      reusing X, y, ids but with w set to LDS weights.

    Args:
        train: DeepChem Dataset for training (required).
        valid: Optional validation Dataset.
        test:  Optional test Dataset.
        lds:   LDSParams controlling bins/kernel/kernel size/sigma.

    Returns:
        (train_with_w, valid_with_w, test_with_w)
    """
    if lds is None:
        lds = LDSParams()

    # Extract train labels and build a frozen weighting function (no leakage).
    y_tr = np.array(train.y).reshape(-1)
    weighter = make_lds_weighter_from_reference(y_tr, lds)

    def _with_weights(ds: Optional[dc.data.Dataset]) -> Optional[dc.data.Dataset]:
        if ds is None:
            return None
        y = np.array(ds.y).reshape(-1)
        w = weighter(y)
        # Preserve shapes DeepChem expects. NumpyDataset will broadcast w if needed.
        return dc.data.NumpyDataset(X=ds.X, y=ds.y, w=w, ids=ds.ids)
    if test!=None:
        return _with_weights(train), _with_weights(valid), _with_weights(test)
    else:
        return _with_weights(train), _with_weights(valid)