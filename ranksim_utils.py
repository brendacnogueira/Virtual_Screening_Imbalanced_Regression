# ranksim_utils.py
import random
import torch
import torch.nn.functional as F


def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))

def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]

class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        r = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, r)
        return r

    @staticmethod
    def backward(ctx, grad_output):
        sequence, r = ctx.saved_tensors
        sequence_prime = sequence + ctx.lambda_val * grad_output
        r_prime = rank_normalised(sequence_prime)
        gradient = -(r - r_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None

def batchwise_ranking_regularizer(features, targets, interp_strength_lambda=1.0, max_batch=256):
    """
    features: (B, d) tensor (already the representation to align)
    targets:  (B, 1) or (B,) tensor of scalar outputs (labels or predictions)
    """
    if features.ndim > 2:
        features = features.view(features.size(0), -1)
    targets = targets.view(-1)

    # Optional capping to keep O(B^2) cost sane
    if features.size(0) > max_batch:
        idx = torch.randperm(features.size(0), device=features.device)[:max_batch]
        x = features[idx]
        y = targets[idx]
    else:
        x = features
        y = targets

    # de-duplicate labels within subset (as in your snippet)
    with torch.no_grad():
        unique_vals = torch.unique(y)
        if unique_vals.numel() < y.numel():
            sel = []
            for v in unique_vals:
                candidates = (y == v).nonzero(as_tuple=False)[:, 0]
                sel.append(candidates[random.randint(0, candidates.numel() - 1)].item())
            sel = torch.tensor(sel, device=y.device, dtype=torch.long)
            x = x[sel]
            y = y[sel]

    x = F.normalize(x, dim=1)
    sim = torch.matmul(x, x.T)  # (n,n)

    loss = 0.0
    n = y.numel()
    for i in range(n):
        label_ranks = rank_normalised((-torch.abs(y[i] - y)).view(1, -1))
        feat_ranks  = TrueRanker.apply(sim[i].view(1, -1), interp_strength_lambda)
        loss = loss + F.mse_loss(feat_ranks, label_ranks)
    return loss / max(1, n)

def ranksim_score_from_predictions(X_np, y_pred_np, lambda_val=1.0, sample_cap=512, device="cpu"):
    """
    RankSim 'metric' for model selection:
    - compare ranks in INPUT FEATURE space vs. ranks in PREDICTION space.
    Lower is better (MSE over rank vectors).
    """
    import numpy as np
    import scipy.sparse as sp
    if sp.issparse(X_np):
        X_np = X_np.toarray()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_pred = torch.tensor(y_pred_np.reshape(-1), dtype=torch.float32, device=device)

    # Use input features as representation proxy
    with torch.no_grad():
        if X.size(0) > sample_cap:
            idx = torch.randperm(X.size(0), device=device)[:sample_cap]
            X = X[idx]
            y_pred = y_pred[idx]
    return float(batchwise_ranking_regularizer(X, y_pred, lambda_val).detach().cpu().item())
# ranksim_utils.py
import torch
import numpy as np

def _make_pytorch_shapes_consistent(y_true, y_pred):
     """
     Minimal drop-in replacement: flattens, casts to float tensors,
     and returns (y_true, y_pred) with consistent shapes for PyTorch ops.
     """
     if isinstance(y_true, np.ndarray):
         y_true = torch.from_numpy(y_true)
     if isinstance(y_pred, np.ndarray):
         y_pred = torch.from_numpy(y_pred)
     y_true = y_true.reshape(-1).float()
     y_pred = y_pred.reshape(-1).float()
     return y_true, y_pred
def make_gcn_ranksim_loss(model_ref, base_dc_loss, alpha=1.0, lambda_val=1.0):
    """
    base_dc_loss: DeepChem Loss (e.g., L1Loss(), L2Loss(), SeraLoss)
    model_ref:    the SAME nn.Module passed to TorchModel (must expose .last_features)
    """
    class _GCNRankSimLoss(Loss):
        def _create_pytorch_loss(self):
            base_loss_fn = base_dc_loss._create_pytorch_loss()
            def loss(output, labels):
                # 1) Compute the base criterion exactly as usual
                output, labels = _make_pytorch_shapes_consistent(output, labels)
                base = base_loss_fn(output, labels)

                # 2) Add RankSim on the cached graph embeddings vs PREDICTIONS
                feats = getattr(model_ref, "last_features", None)
                if feats is None:
                    return base  # no cached embeddings this step

                # Use predictions for ranking signal (detach to avoid 2nd-order)
                preds = output.detach().view(-1)

                rs = batchwise_ranking_regularizer(
                    feats, preds, interp_strength_lambda=lambda_val
                )
                return base + alpha * rs
            return loss
    return _GCNRankSimLoss()
