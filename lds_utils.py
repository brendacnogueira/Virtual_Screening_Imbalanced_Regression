# lds_utils.py
from __future__ import annotations
import numpy as np
from scipy.signal import convolve
from dataclasses import dataclass

def _gaussian_kernel(ks: int, sigma: float) -> np.ndarray:
    c = ks // 2
    x = np.arange(ks) - c
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    return k

def _triang_kernel(ks: int) -> np.ndarray:
    # 1,2,...,c+1,...,2,1 (normalized)
    c = ks // 2
    x = np.arange(ks)
    k = 1 + c - np.abs(x - c)
    k = np.clip(k, 0, None)
    k = k / k.sum()
    return k

def _laplace_kernel(ks: int, sigma: float) -> np.ndarray:
    c = ks // 2
    x = np.arange(ks) - c
    k = np.exp(-np.abs(x) / float(sigma))
    k /= k.sum()
    return k

@dataclass
class LDSParams:
    bins: int = 100
    kernel: str = "gaussian" # {'gaussian','triang','laplace'}
    ks: int = 5 # odd number
    sigma: float = 2.0
    min_density: float = 1e-6 # floor to avoid inf weights


    def kernel_vec(self) -> np.ndarray:
        if self.kernel == "gaussian":
            return _gaussian_kernel(self.ks, self.sigma)
        if self.kernel == "triang":
            return _triang_kernel(self.ks)
        if self.kernel == "laplace":
            return _laplace_kernel(self.ks, self.sigma)
        raise ValueError(f"Unknown kernel: {self.kernel}")

def lds_effective_density(y: np.ndarray, lds: LDSParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LDS effective label density \tilde{p}(y) by (i) estimating the label histogram
    on a fixed grid and (ii) convolving it with the chosen kernel.


    Returns (p, p_tilde, bin_edges), where p and p_tilde are *densities* (sum≈1).
    """
    y = np.asarray(y).ravel()
    p, bin_edges = np.histogram(y, bins=lds.bins, density=True)
    k = lds.kernel_vec()
    p_tilde = convolve(p, k, mode="same")
    # numerical safety
    p_tilde = np.clip(p_tilde, lds.min_density, None)
    return p, p_tilde, bin_edges


def lds_weights(y: np.ndarray, lds: LDSParams) -> np.ndarray:
    """Map each y_i to its convolved density bin and return w_i ∝ 1/\tilde{p}(y_i)."""
    y = np.asarray(y).ravel()
    _, p_tilde, bins = lds_effective_density(y, lds)
    # indices for each y_i
    idx = np.clip(np.digitize(y, bins[:-1]) - 1, 0, len(p_tilde) - 1)
    w = 1.0 / p_tilde[idx]
    # normalize for numerical stability (optional)
    w = w / np.mean(w)
    return w

def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w = np.asarray(w).ravel()
    return float(np.average((y_true - y_pred) ** 2, weights=w))

def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w = np.asarray(w).ravel()
    return float(np.average(np.abs(y_true - y_pred), weights=w))

def lds_mse(y_true: np.ndarray, y_pred: np.ndarray, lds: LDSParams) -> float:
    return weighted_mse(y_true, y_pred, lds_weights(y_true, lds))

def lds_mae(y_true: np.ndarray, y_pred: np.ndarray, lds: LDSParams) -> float:
    return weighted_mae(y_true, y_pred, lds_weights(y_true, lds))

def make_lds_weighter_from_reference(y_ref: np.ndarray, lds: LDSParams):
    """
    Build a frozen LDS-weight function from TRAIN labels (y_ref) ONLY
    to avoid leakage into validation/test.

    Returns a function w(y_new) that maps labels → LDS sample weights
    using the same histogram/binning computed on TRAIN labels.
    """
    y_ref = np.asarray(y_ref).ravel()

    # 1) fit histogram + convolved effective density from TRAIN only
    _, p_tilde, bins = lds_effective_density(y_ref, lds)

    def _weights_new(y_new: np.ndarray) -> np.ndarray:
        y_new = np.asarray(y_new).ravel()

        # Bin indices of new y
        idx = np.clip(np.digitize(y_new, bins[:-1]) - 1, 0, len(p_tilde) - 1)

        # LDS weights  w_i ∝ 1 / p_tilde[y_i]
        w = 1.0 / p_tilde[idx]

        # Normalize to mean 1.0 (not required but stable)
        return w / np.mean(w)

    return _weights_new
