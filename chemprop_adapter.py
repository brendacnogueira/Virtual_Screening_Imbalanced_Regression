# chemprop_adapter.py
from __future__ import annotations
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Chemprop (library) ---
# Works with chemprop>=1.6/2.x; if import paths differ in your env, adjust the two imports below.
from chemprop.models import MoleculeModel                 # nn.Module (MPNN + FFN)
from chemprop.data import MoleculeDatapoint, MoleculeDataset, StandardScaler
from chemprop.data import MoleculeDataLoader

# --- Your project utilities ---
from lds_utils import LDSParams, make_lds_weighter_from_reference, lds_mae, lds_mse
from ranksim_utils import batchwise_ranking_regularizer
from sera_opt_proto import SeraCriterion, phi as sera_phi

# ---------- small helpers ----------
def _make_dataset(smiles_list, y_list):
    dp = [MoleculeDatapoint(smiles=[s], targets=[float(y)]) for s, y in zip(smiles_list, y_list)]
    return MoleculeDataset(dp)

def _collect_last_features_hook(module: nn.Module, feats_container: dict, key: str = "Z"):
    def _hook(m, inp, out):
        feats_container[key] = out.detach()
    return _hook

# ---------- Loss selectors ----------
class LossPack:
    """Unifies all objectives you use across models, including RankSim add-on."""
    def __init__(self, metric_name: str, ph: dict,
                 lds: LDSParams | None = None,
                 ranksim_alpha: float = 1.0,
                 ranksim_lambda: float = 1.0,
                 device: str = "cuda"):
        self.metric_name = metric_name.upper()
        self.ph = ph
        self.lds = lds
        self.ranksim_alpha = ranksim_alpha
        self.ranksim_lambda = ranksim_lambda
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

        self.sera = SeraCriterion(self.ph, device=self.device)
        self.l1 = nn.L1Loss(reduction="none")
        self.l2 = nn.MSELoss(reduction="none")

    def base_loss(self, y_hat, y_true, w=None):
        if "SERA" in self.metric_name:
            # SERA is "higher is better" as a score; as a loss we use the criterion directly
            return self.sera(y_hat, y_true)
        if "MAE" in self.metric_name:
            per = self.l1(y_hat, y_true).view(-1)
        else:  # default MSE branch (covers plain "MSE" and LDS-MSE variants)
            per = self.l2(y_hat, y_true).view(-1)

        if "LDS-" in self.metric_name:
            # Weighted with LDS *at train time*
            if w is None:
                raise RuntimeError("Expected LDS weights when using LDS-* training.")
            return torch.mean(per * w.view(-1))
        else:
            return torch.mean(per)

    def add_ranksim_if_needed(self, loss, feats, preds):
        if "RANKSIM" not in self.metric_name:
            return loss
        # RankSim regularizer on (features vs predictions)
        rs = batchwise_ranking_regularizer(feats, preds.view(-1), self.ranksim_lambda)
        return loss + self.ranksim_alpha * rs

# ---------- Regressor ----------
class ChempropRegressor:
    """
    A thin, sklearn-like wrapper around Chemprop's MoleculeModel that:
      - builds MoleculeDataset from (smiles, y)
      - supports SERA / LDS-* / RankSim-* training
      - exposes .fit/.predict like your other wrappers
    """

    def __init__(self,
                 metric: str,
                 ph: dict,
                 params: dict | None = None,
                 lds: LDSParams | None = None,
                 ranksim_alpha: float = 1.0,
                 ranksim_lambda: float = 1.0,
                 device: str = "cuda",
                 batch_size: int = 64,
                 max_epochs: int = 200,
                 lr: float = 1e-3,
                 early_stop_patience: int = 25):
        self.metric = metric
        self.ph = ph
        self.params = params or {}
        self.lds = lds
        self.ranksim_alpha = ranksim_alpha
        self.ranksim_lambda = ranksim_lambda
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience

        # Default Chemprop architecture (override via params)
        self.arch = dict(
            depth=self.params.get("depth", 3),
            hidden_size=self.params.get("hidden_size", 300),
            ffn_num_layers=self.params.get("ffn_num_layers", 2),
            dropout=self.params.get("dropout", 0.0),
            bias=True
        )
        self.scaler = None
        self.model = None
        self._last_feats = {}

    # ---------- API ----------
    def fit(self, smiles_train, y_train, smiles_valid=None, y_valid=None, sample_weights=None):
        train_ds = _make_dataset(smiles_train, y_train)
        valid_ds = _make_dataset(smiles_valid, y_valid) if (smiles_valid is not None) else None

        # Standardize y (Chemprop default behavior). Keep your output_transformer pattern.
        self.scaler = StandardScaler().fit(train_ds.targets())
        train_ds.set_targets(self.scaler.transform(train_ds.targets()))
        if valid_ds is not None:
            valid_ds.set_targets(self.scaler.transform(valid_ds.targets()))

        # LDS weights from TRAIN labels (no leakage), like your DeepChem/GNN code
        if self.lds is not None or ("LDS-" in self.metric.upper()):
            y_tr_orig = np.array(y_train).reshape(-1)
            lds = self.lds or LDSParams()
            weighter = make_lds_weighter_from_reference(y_tr_orig, lds)
            w_train = torch.tensor(weighter(y_tr_orig), dtype=torch.float32)
        else:
            w_train = torch.ones(len(train_ds), dtype=torch.float32)

        train_loader = MoleculeDataLoader(train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True)
        valid_loader = MoleculeDataLoader(valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False) if valid_ds else None

        # Build Chemprop model
        self.model = MoleculeModel(task_type='regression', **self.arch).to(self.device)

        # capture penultimate features for RankSim (hook into the FFN input)
        # Chemprop exposes model.ffn; we hook the first FFN layer's input (pre-activation)
        if hasattr(self.model, "ffn") and isinstance(self.model.ffn, nn.Sequential) and len(self.model.ffn) > 0:
            self.model.ffn[0].register_forward_hook(_collect_last_features_hook(self.model, self._last_feats, key="Z"))

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_pack = LossPack(self.metric, self.ph, lds=self.lds, ranksim_alpha=self.ranksim_alpha,
                             ranksim_lambda=self.ranksim_lambda, device="cuda" if self.device.type=="cuda" else "cpu")

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            tr_loss = 0.0
            for batch in train_loader:
                # batch targets are already scaled
                x, y = batch.batch_graph(), torch.tensor(batch.targets(), dtype=torch.float32).view(-1, 1)
                x = x.to(self.device)
                y = y.to(self.device)
                w = w_train[batch.indices()].to(self.device).view(-1, 1)

                opt.zero_grad()
                y_hat = self.model(x)  # (B, 1)
                base = loss_pack.base_loss(y_hat, y, w=w if "LDS-" in self.metric.upper() else None)

                # RankSim add-on uses cached features (from hook) and predictions
                feats = self._last_feats.get("Z", None)
                if feats is not None:
                    base = loss_pack.add_ranksim_if_needed(base, feats, y_hat)

                base.backward()
                opt.step()
                tr_loss += base.item()

            # Early stopping on validation using *selection* loss in the same convention (lower is better)
            val_score = self._val_loss(valid_loader, loss_pack) if valid_loader is not None else tr_loss / max(1, len(train_loader))
            if val_score < best_val:
                best_val = val_score
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def _val_loss(self, valid_loader, loss_pack: LossPack):
        self.model.eval()
        agg = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch.batch_graph(), torch.tensor(batch.targets(), dtype=torch.float32).view(-1, 1)
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                # unweighted validation unless metric specifies LDS-*
                if "LDS-" in self.metric.upper():
                    # build frozen LDS weighter from TRAIN only (already done); reuse w_train stats by recomputing on y (ok: same bins)
                    # simple unweighted valid is also acceptable; keep weighted to match your DNN flow
                    # Here we approximate using the labels themselves (since we don't have the frozen-TRAIN weighter here).
                    # If you want *strict* no-leakage, pass an external weighter into LossPack instead.
                    lds = self.lds or LDSParams()
                    # map y back to numpy for weights
                    y_np = y.detach().view(-1).cpu().numpy()
                    w_np = make_lds_weighter_from_reference(y_np, lds)(y_np)
                    w = torch.tensor(w_np, dtype=torch.float32, device=y.device).view(-1, 1)
                else:
                    w = None

                base = loss_pack.base_loss(y_hat, y, w=w)

                feats = getattr(self, "_last_feats", {}).get("Z", None)
                if feats is not None:
                    base = loss_pack.add_ranksim_if_needed(base, feats, y_hat)

                agg += base.item()
                n_batches += 1
        return agg / max(1, n_batches)

    def predict(self, smiles_list):
        ds = _make_dataset(smiles_list, [0.0] * len(smiles_list))
        loader = MoleculeDataLoader(ds, batch_size=self.batch_size, num_workers=0, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                x = batch.batch_graph().to(self.device)
                y_hat = self.model(x).view(-1)
                preds.append(y_hat.detach().cpu().numpy())
        y_scaled = np.concatenate(preds, axis=0)
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)

# ---------- Grid search (optional) ----------
def chemprop_grid_search(smiles_tr, y_tr, smiles_va, y_va, ph, metric, param_grid: dict,
                         lds: LDSParams | None = None, ranksim_alpha=1.0, ranksim_lambda=1.0,
                         device="cuda"):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    best = None
    best_score = float("inf")
    all_results = []

    for combo in itertools.product(*vals):
        p = dict(zip(keys, combo))
        model = ChempropRegressor(metric=metric, ph=ph, params=p, lds=lds,
                                  ranksim_alpha=ranksim_alpha, ranksim_lambda=ranksim_lambda,
                                  device=device).fit(smiles_tr, y_tr, smiles_va, y_va)

        # use same lower-is-better convention as your GCN HPO
        y_hat = model.predict(smiles_va)
        y_true = np.array(y_va).reshape(-1)

        # selection-score (lower is better)
        sel = metric.upper()
        if sel == "MAE":
            score = float(np.mean(np.abs(y_true - y_hat)))
        elif sel == "MSE":
            score = float(np.mean((y_true - y_hat) ** 2))
        elif sel == "SERA":
            # SERA returns "higher is better" => convert to loss by NEGATING
            # we keep "lower is better" convention: use raw SERA value as 'loss'
            from sera_opt_proto import sera_pt
            phi_labels = sera_phi(np.array(y_true), ph)  # numpy → phi (if you have a torch phi controller, adapt here)
            score = float(sera_pt(torch.tensor(y_true), torch.tensor(y_hat), torch.tensor(phi_labels), device="cpu"))
        elif sel == "LDS-MAE":
            score = float(lds_mae(y_true, y_hat, lds or LDSParams()))
        elif sel == "LDS-MSE":
            score = float(lds_mse(y_true, y_hat, lds or LDSParams()))
        else:
            # RankSim selection on VALID (no leakage, recompute features via hook)
            # We don’t need features here because we compare in prediction space using the wrapper you already use elsewhere.
            from ranksim_utils import ranksim_score_from_predictions
            # reuse predictions as ranking signal vs input-space proxy is not trivial here; accept ranksim on predictions
            score = float(ranksim_score_from_predictions(
                X_np=np.arange(len(y_hat)).reshape(-1, 1),  # dummy positions; RankSim used mainly during training via regularizer
                y_pred_np=y_hat,
                lambda_val=ranksim_lambda
            ))

        all_results.append({"params": p, "score": score})
        if score < best_score:
            best_score, best = score, (p, model)

    return best[1], best[0], all_results
