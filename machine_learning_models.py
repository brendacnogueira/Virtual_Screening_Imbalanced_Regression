import itertools
from typing import *
import numpy as np
import pandas as pd
import os
import random
import math

# Tensorflow/Keras
import keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Lambda, Input
from tensorflow.python.ops import math_ops

# Sklearn
from sklearn import neighbors, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from ml_utils import tanimoto_from_sparse #, untransform_data

import warnings

#torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

#XGboost
import xgboost as xgb
from sklearn.metrics import make_scorer
from lds_utils import LDSParams, lds_weights, lds_mse, lds_mae, make_lds_weighter_from_reference

from ranksim_utils import ranksim_score_from_predictions  # NEW
from ranksim_utils import batchwise_ranking_regularizer

###############################ADDTIONAL FOR SERA#############################
#import SERA
import torch
from torch.utils.data import DataLoader
from sera_opt_proto import phi,sera_pt,SeraCriterion
from sklearn.metrics import make_scorer
from xgboost_sera import sera_loss

import deepchem as dc

os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"


def SERA_opt(y_true,y_pred,ph,device="cpu"):
    phi_labels=SERA_phi_control(y_true,ph)
    sera=sera_pt(torch.Tensor(y_true),torch.Tensor(y_pred),phi_labels,device=device)
    return float(np.array(sera))

def SERA_phi_control(y,ph,out_dtype_np=np.float32):
    #y = np.float64(np.array(y))
    #phi_values = phi(pd.Series(y.flatten()), phi_parms=ph)
    phi_values=phi(pd.Series(y.flatten(),dtype="float64"), phi_parms=ph)
    return torch.Tensor(np.array(phi_values, dtype=out_dtype_np))


# --- Balanced MSE (BMC) -------------------------------------------------------
import torch.nn.functional as F

def bmc_loss(pred, target, noise_var):
    """
    pred:   (B, 1) tensor
    target: (B, 1) tensor
    noise_var: float or (1,) tensor with sigma^2
    """
    # pred-target^T: (B, B)
    logits = - (pred - target.t()).pow(2) / (2 * noise_var)
    # targets are indices 0..B-1 (diagonal = matching pairs)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))
    # Optional: restore MSE scale
    loss = loss * (2 * noise_var).detach()
    return loss

class BMCLoss(nn.Module):
    """
    Balanced MSE with learnable sigma.
    init_noise_sigma: float (e.g., 0.5). We optimize sigma during training.
    """
    def __init__(self, init_noise_sigma: float = 0.5):
        super().__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(float(init_noise_sigma)))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def _bmse_scorer_fn(y_true, y_pred, init_sigma: float = 0.5):
    # y_true, y_pred are 1D numpy arrays
    yt = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1)
    yp = torch.tensor(y_pred, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        loss = bmc_loss(yp, yt, noise_var=torch.tensor(init_sigma**2))
    # sklearn scorers are "higher is better"; we want to minimize loss
    return -float(loss.item())

def _ranksim_scorer(estimator, X, y_true):
        # predict then compare ranks in feature-space vs predictions
        y_hat = estimator.predict(X)
        return -ranksim_score_from_predictions(X, y_hat, lambda_val=1.0)  # negate so "smaller is better" -> "larger is better" for sklearn
###################################################################################

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

class Dataset:
    def __init__(self, features: np.array, labels: np.array):
        self.features = features
        self.labels = labels
        self._add_instances = set()

    def add_instance(self, name, values: np.array):
        self._add_instances.add(name)
        self.__dict__[name] = values

    @property
    def columns(self) -> dict:
        data_dict = {k: v for k, v in self.__dict__.items()}
        data_dict['features'] = self.features
        data_dict['labels'] = self.labels
        return data_dict

    def __len__(self):
        return self.labels.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}
        
        subset = Dataset(self.features[idx], self.labels[idx])
        for addt_instance in self._add_instances:
            subset.add_instance(addt_instance, self.__dict__[addt_instance][idx])

        return subset

class WeightedKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, metric='minkowski', p=2):
        """
        Initialize the WeightedKNNRegressor.

        Parameters:
        - n_neighbors (int): Number of neighbors to consider.
        - metric (str): Distance metric to use. Default is 'minkowski'.
        - p (int): Parameter for Minkowski metric (e.g., p=2 for Euclidean distance).
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.neigh = None
        self.X_train = None
        self.y_train = None
        self.sample_weights = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the WeightedKNNRegressor model.

        Parameters:
        - X (array-like): Training data of shape (n_samples, n_features).
        - y (array-like): Target values of shape (n_samples,).
        - sample_weight (array-like): Weights for each sample. Default is None.
        """
        self.neigh = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)
        self.neigh.fit(X)
        self.X_train = X
        self.y_train = y
        self.sample_weights = sample_weight if sample_weight is not None else np.ones(len(y))
        return self

    def predict(self, X):
        """
        Predict target values for given data.

        Parameters:
        - X (array-like): Test data of shape (n_samples, n_features).

        Returns:
        - predictions (array): Predicted target values of shape (n_samples,).
        """
        distances, indices = self.neigh.kneighbors(X)

        predictions = []
        for i, neighbors in enumerate(indices):
            neighbor_distances = distances[i]
            neighbor_weights = self.sample_weights[neighbors]
            
            # Use inverse distance weighting, adjusting with sample weights
            if np.any(neighbor_distances == 0):
                # Avoid division by zero; if a point has distance 0, use its value directly
                zero_dist_idx = np.where(neighbor_distances == 0)[0][0]
                predictions.append(self.y_train[neighbors[zero_dist_idx]])
            else:
                # Compute weighted average
                weights = neighbor_weights / (neighbor_distances + 1e-9)  # Avoid division by zero
                predictions.append(np.dot(weights, self.y_train[neighbors]) / weights.sum())

        return np.array(predictions)

    def get_params(self, deep=True):
        """
        Get the parameters of the WeightedKNNRegressor as a dictionary.

        Parameters:
        - deep (bool): If True, return parameters for this estimator and contained subobjects.
        
        Returns:
        - params (dict): Dictionary of model parameters.
        """
        return {'n_neighbors': self.n_neighbors, 'metric': self.metric, 'p': self.p}

    def set_params(self, **params):
        """
        Set the parameters of the WeightedKNNRegressor.

        Parameters:
        - params (dict): Dictionary of parameters to set.
        
        Returns:
        - self: The updated estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class MLModel:
    def __init__(self, data, ml_algorithm, metric, ph, weights=None,
                 parameters='grid', cv_fold=10, random_seed=2002):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.metric=metric
        self.ph=ph
        self.weights=weights
       
        self.cv_fold = cv_fold
        self.seed = random_seed
        self.parameters = parameters
        self.h_parameters = self.hyperparameters()
        self.model, self.cv_results = self.cross_validation()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()
        

    def hyperparameters(self):
        if self.parameters == "grid":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 10000],
                            'kernel': [tanimoto_from_sparse],
                            }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [25, 100, 200],
                                'max_features': ['auto'],
                                'min_samples_split': [2, 3, 5],
                                'min_samples_leaf': [1, 2, 5],
                                }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [1, 3, 5]
                            }
                elif self.ml_algorithm=="XGBoost":
                     return {'n_estimators': [10, 50, 100, 250], 
                             'learning_rate': [.001, .01, .1], 
                             'max_depth': [3, 5, 10],
                             }



    def cross_validation(self):

        
        sera_score=make_scorer(SERA_opt,greater_is_better=False,ph=self.ph)
        
        lds = LDSParams() # default: gaussian, ks=5, sigma=2

       
        

        metrics_dict = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "SERA": sera_score,
        # NEW (optimization objectives):
        "LDS-MSE": make_scorer(lds_mse, greater_is_better=False, lds=lds),
        "LDS-MAE": make_scorer(lds_mae, greater_is_better=False, lds=lds),
        "RANKSIM": _ranksim_scorer,
        "BMSE": make_scorer(_bmse_scorer_fn, greater_is_better=True), 
        }
        if "MSE" in self.metric:
             opt_metric = metrics_dict["MSE"]
        elif "MAE" in self.metric:
            opt_metric = metrics_dict["MAE"]
        else:
           
            opt_metric = metrics_dict[self.metric]
        
        if self.ml_algorithm == "MR":
            model = DummyRegressor()
        elif self.ml_algorithm == "SVR":
            model = SVR()
        elif self.ml_algorithm == "RFR":
            model = RandomForestRegressor(random_state=self.seed)
        elif self.ml_algorithm == "kNN":
            
            model = WeightedKNNRegressor()
        elif self.ml_algorithm=="XGBoost":
            loss_xgboost = {
                "MAE":      "reg:absoluteerror",
                "MSE":      "reg:squarederror",
                "SERA":     sera_loss(self.ph),       # your existing custom objective
                
            }
            if "MSE" in self.metric:
                loss=loss_xgboost["MSE"]
            elif "MAE" in self.metric:
                loss=loss_xgboost["MAE"]
            else:
                loss=loss_xgboost[self.metric]

            model=xgb.XGBRegressor(random_state=self.seed, objective=loss)

        cv_results = GridSearchCV(model,
                                  param_grid=self.h_parameters,
                                  cv=self.cv_fold,
                                  scoring=opt_metric,
                                  n_jobs=-1,
                                )
        fit_kwargs = {}
        if "LDS" in self.metric:
            w = lds_weights(self.data.labels, lds)
            fit_kwargs["sample_weight"] = w
        else:
            if self.weights is not None:
                fit_kwargs["sample_weight"] = self.weights


        cv_results.fit(self.data.features, self.data.labels, **fit_kwargs)

        
        return model, cv_results

    def optimal_parameters(self):
        best_params = self.cv_results.cv_results_['params'][self.cv_results.best_index_]
        return best_params

    def final_model(self):
        model = self.model.set_params(**self.best_params)
        return model.fit(self.data.features, self.data.labels)


import copy   
class EarlyStopping_torch:
    def __init__(self, patience=50, min_delta=0, restore_best_weights=False):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

class WeightedMSELoss(nn.Module):
    """
    Custom Weighted Mean Squared Error Loss.

    Parameters:
    - weights: Tensor of weights for each sample.
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions, targets, weights):
        """
        Compute the weighted MSE loss.

        Parameters:
        - predictions (Tensor): Predicted values, shape (n_samples,).
        - targets (Tensor): Ground truth values, shape (n_samples,).
        - weights (Tensor): Sample weights, shape (n_samples,).

        Returns:
        - loss (Tensor): Weighted MSE loss.
        """
        # Compute squared differences
        squared_diff = (predictions - targets) ** 2
        
        # Apply weights to the squared differences
        weighted_squared_diff = weights * squared_diff

        # Compute the mean of the weighted squared differences
        loss = torch.mean(weighted_squared_diff)
        return loss
    
class WeightedMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, predictions, targets, weights):
        return torch.mean(weights * torch.abs(predictions - targets))


import scipy.sparse as sp
def _to_dense_numpy(X):
    """Return dense float32 numpy array for dense or scipy.sparse inputs."""
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)

class DNN_torch(torch.nn.Module):
    def __init__(self,layers=None, dropout_rate= 0, activation='tanh',
                 n_features=None, seed= None, reg_class="regression"):
        # Init parent
        super(DNN_torch, self).__init__()
        torch.manual_seed(seed) 
        
        self.layers=layers
        self.activation=F.tanh if activation=='tanh' else F.relu
        self.dropout=nn.Dropout(p=dropout_rate)
        

        if layers is None:
            layers = (100, 100)
        
        self.list_layers=nn.ModuleList()
        
        for i, net_nodes in enumerate(layers, 0):
            if i == 0:
                self.list_layers.append(nn.Linear(n_features,net_nodes))
            else:
                self.list_layers.append(nn.Linear(layers[i-1],net_nodes))

        if reg_class == "regression":  
                self.list_layers.append(nn.Linear(layers[len(layers)-1],1))
    
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.list_layers, 1):
            last = (i == len(self.list_layers))
            if last:
                out = layer(h)
            else:
                h = self.activation(layer(h))
                h = self.dropout(h)
        # Save penultimate activations as "features"
        self.last_features = h
        return out
            
class DNN:
    
    def __init__(self, data, ml_algorithm, n_features, seed, ph, metric="SERA", weights=None,
                 reg_class="regression", parameters='grid', device="cuda",
                 ranksim_alpha=1.0, ranksim_lambda=1.0,
                 bmse_init_sigma: float = 0.5, bmse_sigma_lr: float = 0.05):
        
        self.bmse_init_sigma = bmse_init_sigma
        self.bmse_sigma_lr   = bmse_sigma_lr

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.ph=ph
        self.metric=metric
        self.weights=weights
        if self.weights is None:
           self.weights=np.ones(len(self.data.labels))
        self.n_features = n_features
        self.seed = seed
        self.reg_class = reg_class
        self.parameters = parameters
        self.device = device
        self.ranksim_alpha = ranksim_alpha
        self.ranksim_lambda = ranksim_lambda
        self.h_parameters = self.dnn_hyperparameters()
        self.cv_results = self.dnn_cross_validation()
        self.best_params = self.dnn_select_best_parameters()
        self.model = self.final_model()
        

    
    def dnn_hyperparameters(self):
        if self.parameters == "grid":

            return {
                "module__layers": [(100, 100), (250, 250), (250, 500), (500, 250), (500, 250, 100), (100, 250, 500)],
                "module__dropout_rate": [0.0, 0.25, 0.5],
                "module__activation": ['tanh'],
                "optimizer__lr": [0.1, 0.01, 0.001],
                "max_epochs": [200]
            }

        
    def dnn_train(self, features_train, labels_train, features_valid, labels_valid,
              nn_layers, dropout_rate, activation, learning_rate, train_epochs,
              weights_train, weights_valid):

        train_dataset = list(zip(features_train, labels_train, weights_train))
        loader_train = DataLoader(train_dataset, shuffle=True, batch_size=32)

        if features_valid is not None:
            val_dataset = list(zip(features_valid, labels_valid, weights_valid))
            loader_val = DataLoader(val_dataset, shuffle=False, batch_size=32)

        # device
        use_cuda = (self.device == "cuda" and torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")
        if not use_cuda:
            print("Running on CPU")
        

        model = DNN_torch(
            layers=nn_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            n_features=self.n_features,
            seed=self.seed
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if metric_name == "BMSE":
                    optimizer.add_param_group({
                        "params": [criterion.noise_sigma],
                        "lr": max(1e-5, float(self.bmse_sigma_lr) * float(learning_rate)),
                        "name": "noise_sigma"
                    })
                

        loss_dict = {
            "WMSE": WeightedMSELoss(),
            "WMAE": WeightedMAELoss(),
            "MSE": torch.nn.MSELoss(),
            "MAE": torch.nn.L1Loss(),
            "SERA": SeraCriterion(self.ph, device=device),
            "BMSE": BMCLoss(init_noise_sigma=self.bmse_init_sigma),
        }

        metric_name = str(self.metric).upper()
       
        if metric_name == "LDS-MSE":
            criterion = loss_dict["WMSE"]
        elif metric_name == "LDS-MAE":
            criterion = loss_dict["WMAE"]
        elif "MSE" in metric_name:
            criterion = loss_dict["MSE"]
        elif "MAE" in metric_name:
            criterion = loss_dict["MAE"]
        elif "BMSE" in metric_name:
            criterion = loss_dict["BMSE"]
        else:
            criterion = loss_dict.get(self.metric, torch.nn.MSELoss())
        

        es = EarlyStopping_torch()
        done = False
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(train_epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch, w_batch in loader_train:
                X_batch = X_batch.to(device, dtype=torch.float32, non_blocking=use_cuda)
                y_batch = y_batch.to(device, dtype=torch.float32, non_blocking=use_cuda).reshape(-1, 1)
                w_batch = w_batch.to(device, dtype=torch.float32, non_blocking=use_cuda).reshape(-1, 1)

                optimizer.zero_grad()
                pred = model(X_batch).reshape(-1, 1)

                if metric_name in {"LDS-MSE", "LDS-MAE", "WMSE", "WMAE"}:
                    loss = criterion(pred, y_batch, w_batch)
                else:
                    loss = criterion(pred, y_batch)
                if "RANKSIM" in metric_name:
                    feats = model.last_features  # (B, d)
                    rs = batchwise_ranking_regularizer(feats, y_batch.view(-1), self.ranksim_lambda)
                    loss = loss + self.ranksim_alpha * rs
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(1, len(loader_train))
            history["train_loss"].append(train_loss)

            # validation
            if features_valid is not None:
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for Xb, yb, wb in loader_val:
                        Xb = Xb.to(device, dtype=torch.float32)
                        yb = yb.to(device, dtype=torch.float32)
                        wb = wb.to(device, dtype=torch.float32)
                        Y_hat = model(Xb)
                        if metric_name in {"LDS-MSE", "LDS-MAE", "WMSE", "WMAE"}:
                            l = criterion(Y_hat, yb, wb)
                        else:
                            l = criterion(Y_hat, yb)

                        if metric_name=="RANKSIM":
                            feats = model.last_features  # (B, d)
                            rs = batchwise_ranking_regularizer(feats, yb.view(-1), self.ranksim_lambda)
                            l = loss + self.ranksim_alpha * rs

                        valid_loss += l.item()


                valid_loss /= max(1, len(loader_val))
                history["val_loss"].append(valid_loss)

                if es(model, valid_loss):
                    done = True
            else:
                if es(model, train_loss):
                    done = True

            if done:
                break

        return history, model

    def dnn_cross_validation(self):
        features = self.data.features
        labels   = self.data.labels
        base_w   = self.weights  # may be None

        X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
            features, labels, base_w, random_state=self.seed
        )

        # ---- densify ----
        X_tr = _to_dense_numpy(X_tr)
        X_va = _to_dense_numpy(X_va)

        # ---- LDS or provided weights ----
        metric_name = str(self.metric).upper()
        if metric_name in {"LDS-MSE", "LDS-MAE"}:
            lds = LDSParams(bins=100, kernel="gaussian", ks=5, sigma=2.0)
            # Build frozen weighter from TRAIN labels (no leakage)
            weighter = make_lds_weighter_from_reference(y_tr, lds)
            w_tr_np = weighter(y_tr)
            # Option A (recommended): also weight validation so val loss reflects LDS
            w_va_np = weighter(y_va)
            # Option B: unweighted valid (comment previous line, use ones)
            # w_va_np = np.ones_like(y_va, dtype=np.float32)
        else:
            # Use given weights or default ones
            if w_tr is None:
                w_tr_np = np.ones_like(y_tr, dtype=np.float32)
                w_va_np = np.ones_like(y_va, dtype=np.float32)
            else:
                w_tr_np = np.asarray(w_tr, dtype=np.float32).reshape(-1)
                w_va_np = np.asarray(w_va, dtype=np.float32).reshape(-1)

        # ---- to tensors with correct shapes ----
        features_train = torch.from_numpy(X_tr)                                 # (Ntr, d)
        labels_train   = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).reshape(-1, 1)
        features_valid = torch.from_numpy(X_va)                                 # (Nva, d)
        labels_valid   = torch.from_numpy(np.asarray(y_va, dtype=np.float32)).reshape(-1, 1)
        weights_train  = torch.from_numpy(w_tr_np.astype(np.float32)).reshape(-1, 1)
        weights_valid  = torch.from_numpy(w_va_np.astype(np.float32)).reshape(-1, 1)

        # ---- grid search ----
        hyperparameters = self.h_parameters
        parameter_grid = itertools.product(*hyperparameters.values())

        grid_search_results = dict()
        for i, grid_comb in enumerate(parameter_grid):
            nn_layers, dropout_rate, activation, learning_rate, train_epochs = grid_comb
            history, _ = self.dnn_train(
                features_train, labels_train,
                features_valid, labels_valid,
                nn_layers, dropout_rate, activation, learning_rate, train_epochs,
                weights_train, weights_valid
            )
            grid_search_results[grid_comb] = history

        return grid_search_results

            
    def dnn_select_best_parameters(self):

        """
        Grid Search selection of best parameters

        """
        grid_search_results = []

        for param_comb, fitted_model in self.cv_results.items():
            # Model Training history
            history_data = pd.DataFrame(fitted_model)
            
            # importing package 
            
            # Save stopping epoch
            history_data = history_data.reset_index().rename(columns={"index": "on_epoch"})
            history_data["on_epoch"] = history_data["on_epoch"].apply(lambda x: x + 1)
 
            # Select loss with minimum validation loss
            best_per_model = history_data.loc[history_data["val_loss"].idxmin(skipna=True)].rename(param_comb)
            
            grid_search_results.append(best_per_model)

        # Concatenate all models training results
        grid_search_results = pd.concat(grid_search_results, axis=1).T

        # select optimal hyperparameter settings
        optimal_stats = grid_search_results.loc[grid_search_results["val_loss"].idxmin(skipna=True)]
        opt_setting = {k: v for k, v in zip(self.h_parameters.keys(), optimal_stats.name)}
       
        opt_setting.update(optimal_stats.to_dict())

        return opt_setting

    def final_model(self):
        X = _to_dense_numpy(self.data.features)
        y = np.asarray(self.data.labels, dtype=np.float32).reshape(-1)
        base_w = self.weights

        metric_name = str(self.metric).upper()
        if metric_name in {"LDS-MSE", "LDS-MAE"}:
            lds = LDSParams(bins=100, kernel="gaussian", ks=5, sigma=2.0)
            weighter = make_lds_weighter_from_reference(y, lds)
            w = weighter(y)
        else:
            if base_w is None:
                w = np.ones_like(y, dtype=np.float32)
            else:
                w = np.asarray(base_w, dtype=np.float32).reshape(-1)

        features = torch.from_numpy(X.astype(np.float32))
        labels   = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
        weights  = torch.from_numpy(w.astype(np.float32)).reshape(-1, 1)

        _, best_model = self.dnn_train(
            features, labels,
            None, None,
            self.best_params["module__layers"],
            self.best_params["module__dropout_rate"],
            self.best_params["module__activation"],
            self.best_params["optimizer__lr"],
            int(self.best_params["on_epoch"]),
            weights, None
        )
        return best_model




from ml_utils import SearchModel, MultivariateGaussian
from ml_utils import get_data_below_percentile
import oracles
import mbo
class Oracle_af:
    def __init__(self, data, cv_fold=10,seed=2002):
        self.data = data
        
        percentile = 80
        
       
        #Xtrain_nxm, gttrain_n, _ = get_data_below_percentile(data.features, data.labels, percentile, seed=seed)
        #gt_sigma = 1.0
        #n_train = gttrain_n.size
        
        init_searchmodel = MultivariateGaussian(data.features.shape[1])
        init_searchmodel.fit(data.features)
        
        _, input_dim =data.features.shape
        n_neuralnets = 3 #3
        hidden_units = (100, 100, 100, 100, 10)
        oracle = oracles.DeepEnsemble(input_dim=input_dim, n_nn=n_neuralnets, hidden_units=hidden_units)
        oracle.fit(data.features, data.labels)

        mbo_hp=0.9
        
        n_iter = 20 #20 
        iw_alpha = 0.2
        designalgaf = mbo.ConditioningByAdaptiveSampling(mbo_hp)
        self.weights= designalgaf.run(data.features, data.labels, None, oracle, init_searchmodel, autofocus=True,
                                  iw_alpha=iw_alpha, n_iter=n_iter)
        
    


class Model_Evaluation:
    def __init__(self, model, data, data_idx, metric, ph, smiles=None, data_label=None, data_transformer=None, model_id=None,device="cuda"):
       
        self.model_id = model_id
        self.metric=metric
        self.data_transformer = data_transformer
        self.smiles=smiles
        self.model = model
        self.data = data
        self.ph=ph
        self.data_idx=data_idx
        self.data_labels = data_label
        self.device=device
        self.labels, self.y_pred, self.predictions = self.model_predict(data)
        self.pred_performance = self.prediction_performance(data)

    def model_predict(self, data):
        """
        data: should carry .features, .labels, and optionally .ids/.smiles/.target
        """
        def _get_untransformed_labels(dc_dataset):
            # Prefer labels explicitly provided to Model_Evaluation (already untransformed)
            if getattr(self, "data_labels", None) is not None:
                return np.asarray(self.data_labels).ravel()
            # Otherwise pull labels from the DC dataset and untransform them to original scale
            if hasattr(dc_dataset, "y") and dc_dataset.y is not None:
                return np.asarray(self.data_transformer.untransform(dc_dataset.y)).ravel()
            raise ValueError("Ground-truth labels not available: provide `data_labels` or ensure dataset has `.y`.")
        # ---- GCN (DeepChem) path ----
        if self.model_id == 'GCN':
            # DeepChem models expect a DC Dataset in .predict; returns numpy array
            raw_pred = self.model.predict(data).flatten()
            y_prediction = self.data_transformer.untransform(raw_pred)
            labels = _get_untransformed_labels(data)  # use labels from 'data' passed in

        # ---- DNN (PyTorch) path ----
        elif self.model.ml_algorithm == "DNN":
            with torch.no_grad():
                # Densify before torch
                X_np = _to_dense_numpy(data.features)
                X_t  = torch.from_numpy(X_np).float()

                use_cuda = (self.device == "cuda" and torch.cuda.is_available())
                device = torch.device("cuda" if use_cuda else "cpu")
                model = self.model.model.to(device)
                model.eval()

                if use_cuda:
                    Y_hat = model(X_t.cuda())
                    y_prediction = Y_hat.flatten().detach().cpu().numpy()
                else:
                    Y_hat = model(X_t)
                    y_prediction = Y_hat.flatten().detach().numpy()

                labels = np.asarray(data.labels).ravel()

        # ---- sklearn / other models ----
        else:
            X = data.features
            # Most sklearn regressors accept sparse; if yours doesn’t, densify
            try:
                y_prediction = self.model.model.predict(X).ravel()
            except TypeError:
                # e.g., if estimator needs dense
                y_prediction = self.model.model.predict(_to_dense_numpy(X)).ravel()
            labels = np.asarray(data.labels).ravel()

        # ---- assemble predictions DataFrame ----
        predictions = pd.DataFrame(
            {"true": labels, "predicted": y_prediction}
        )

        if self.model_id == 'GCN':
            # DeepChem datasets usually have .ids
            if hasattr(data, "ids"):
                predictions['Target ID'] = data.ids
            predictions['algorithm'] = self.model_id
            predictions['MetricOpt'] = self.metric
            if hasattr(self, "smiles"):
                predictions['Molecules'] = self.smiles
            if hasattr(self, "data_idx"):
                predictions['Index'] = self.data_idx
        else:
            # Your non-GCN metadata layout
            tgt = None
            if hasattr(data, "target"):
                try:
                    tgt = data.target[0]
                except Exception:
                    tgt = data.target
            predictions['Target ID'] = tgt
            predictions['algorithm'] = self.model.ml_algorithm
            predictions['MetricOpt'] = self.metric
            if hasattr(data, "smiles"):
                predictions['Molecules'] = data.smiles
            if hasattr(self, "data_idx"):
                predictions['Index'] = self.data_idx

        return labels, y_prediction, predictions

    def prediction_performance(self, data, y_score=None, nantozero=False) -> pd.DataFrame:

    
        labels = self.labels
        pred = self.y_pred

        fill = 0 if nantozero else np.nan
        if len(pred) == 0:
            mae = fill
            mse = fill
            rmse = fill
            r2 = fill
            sera=fill

        else:
            mae = mean_absolute_error(labels, pred)
            mse = metrics.mean_squared_error(labels, pred)
            rmse = metrics.mean_squared_error(labels, pred, squared=False)
            r2 = metrics.r2_score(labels, pred)
            
            

        if self.model_id == 'GCN':
            target =data.ids[0] # data[0].id
            model_name = self.model_id
        else:
            target = data.target[0]
            model_name = self.model.ml_algorithm

        result_dict = {"MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2": r2,
                        "data_set_size": len(labels),
                        "Target ID": target,
                        "Algorithm": model_name,
                        "Selection Metric": self.metric}
                        
        
        
        
    
        #ph={"method":"range","npts": 3,"control_pts":[0,0,0,8,0.1,0,9,1,0]}
        phi_labels=SERA_phi_control(labels,self.ph)
        sera=sera_pt(torch.Tensor(labels),torch.Tensor(pred),phi_labels)/len(labels)
        result_dict["SERA"]=float(np.array(sera))

        result_list=[result_dict]

        # Prepare result dataset
        results = pd.DataFrame(result_list)
        results = results[["Target ID", "Algorithm", "Selection Metric","MAE", "MSE", "RMSE", "R2","SERA"]]
        results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
        results.set_index(["Target ID", "Algorithm", "Selection Metric"], inplace=True)
        results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2","SERA"]],
                                                        names=["Value", "Metric"])
        results = results.stack().reset_index().set_index("Target ID")
        
        return results


