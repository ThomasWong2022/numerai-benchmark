#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of benchmark models for temporal tabular data
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pandas as pd
import numpy as np
import joblib
import shutil
import os

from .util import align_features_target, RollingTSTransformer, GroupedTimeSeriesSplit
from .feature import benchmark_features_transform


from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit


## Machine Learning packages
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

## Pytorch_tabular
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
        ExperimentConfig,
    )

    from pytorch_tabular.models import (
        CategoryEmbeddingModelConfig,
        AutoIntConfig,
    )
    from pytorch_tabular.models import (
        TabTransformerConfig,
        NodeConfig,
    )
except:
    pass


import lightgbm
import xgboost
import catboost
import pytorch_tabnet


### PyTorch Lightning

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything


class MLP(pl.LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config = config

        neuron_sizes = config.get("neurons", [256, 256])

        self.layers = nn.Sequential(
            nn.Linear(config["input_shape"], neuron_sizes[0]),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.5)),
        )

        for i in range(
            1,
            len(neuron_sizes),
        ):
            self.layers.append(
                nn.Linear(neuron_sizes[i - 1], neuron_sizes[i]),
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config.get("dropout", 0.5)))

        self.layers.append(nn.Linear(neuron_sizes[-1], config["output_shape"]))

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x.float())
        loss = F.mse_loss(y_hat, y.float())
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class NumeraiMLP:
    def __init__(self, config):

        self.config = config
        seed_everything(config.get("seed", 0), workers=True)

    def train(self, X_train, y_train, X_validate, y_validate):

        self.config["input_shape"] = X_train.shape[1]
        self.config["output_shape"] = y_train.shape[1]

        self.network = MLP(self.config)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=self.config.get("patience", 10),
            verbose=False,
            mode="min",
        )

        ## Assume X is a DataFrame, assume y is a DataFrame or pd Series
        dataset_train = TensorDataset(
            torch.from_numpy(X_train.values), torch.from_numpy(y_train.values)
        )
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.config.get("batch_size", 4096)
        )
        dataset_validate = TensorDataset(
            torch.from_numpy(X_validate.values), torch.from_numpy(y_validate.values)
        )
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=self.config.get("batch_size", 4096)
        )

        ## Use GPU if possible
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            deterministic=True,
            auto_lr_find=True,
            max_epochs=self.config.get("max_epochs", 200),
            callbacks=[early_stop_callback],
        )

        self.trainer.fit(self.network, dataloader_train, dataloader_validate)

    def predict(self, X):
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(torch.from_numpy(X.astype(np.float32).values))
        return predictions.numpy()

    def load_model(self, checkpoint):
        self.network = MLP.load_from_checkpoint(checkpoint)

    def save_model(self, checkpoint):
        self.trainer.save_checkpoint(checkpoint)


### Persistence of ML models

### Save Best Model using method provided
def save_best_model(model, model_type, outputpath):
    if model_type in [
        "lightgbm",  ## Backward Comptabile
        "lightgbm-gbdt",
        "lightgbm-goss",
        "lightgbm-rf",
        "lightgbm-dart",
    ]:
        model.booster_.save_model(outputpath)
    if model_type in [
        "xgboost",  ## Backward Comptabile
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        model.save_model(outputpath)
    if model_type == "catboost":
        model.save_model(outputpath)
    if model_type == "tabnet":
        model.save_model(outputpath)
        os.rename("{}.zip".format(outputpath), outputpath)
    if model_type in [
        "pytorch-tabular",  ## Backward Comptabile
        "pytorch-tabular-tabtransformer",
        "pytorch-tabular-node",
        "pytorch-tabular-categoryembedding",
    ]:
        ## Save at a folder
        model.save_model(outputpath)
    if model_type in [
        "Numerai-MLP",
    ]:
        model.save_model(outputpath)
    return None


### load Best Model using method provided
def load_best_model(model_type, outputpath):
    if model_type in [
        "lightgbm",  ## Backward Comptabile
        "lightgbm-gbdt",
        "lightgbm-goss",
        "lightgbm-rf",
        "lightgbm-dart",
    ]:
        reg = lightgbm.Booster(model_file=outputpath)
    if model_type in [
        "xgboost",  ## Backward Comptabile
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        reg = xgboost.XGBRegressor()
        reg.load_model(outputpath)
    if model_type == "catboost":
        reg = catboost.CatBoost()
        reg.load_model(outputpath)
    if model_type == "tabnet":
        reg = pytorch_tabnet.tab_model.TabNetRegressor()
        reg.load_model(outputpath)
    if model_type in [
        "pytorch-tabular",  ## Backward Comptabile
        "pytorch-tabular-tabtransformer",
        "pytorch-tabular-node",
        "pytorch-tabular-categoryembedding",
    ]:
        ## Save at a folder
        from pytorch_tabular import TabularModel

        reg = TabularModel.load_from_checkpoint(outputpath)
    if model_type in [
        "Numerai-MLP",
    ]:
        reg = NumeraiMLP(config=dict())
        reg.load_model(outputpath)
    return reg


### Fit ML Models


def benchmark_tree_model(
    extracted_features_train,
    y_train,
    weights_train,
    extracted_features_test=None,
    y_test=None,
    weights_test=None,
    tabular_model="lightgbn-gbdt",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
):

    ### Lightbgm
    if tabular_model in [
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
        "lightgbm-rf",
    ]:
        if tabular_hyper:
            if tabular_hyper.get("objective", "regression") == "cross_entropy":
                reg = LGBMClassifier(**tabular_hyper)
            else:
                reg = LGBMRegressor(**tabular_hyper)
        else:
            tabular_hyper = {
                "n_estimators": 150,
                "max_depth": 5,
                "random_seed": 0,
                "n_jobs": -1,
            }
            reg = LGBMRegressor(**tabular_hyper)

    ### XGBoost
    if tabular_model in [
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        if tabular_hyper:
            reg = XGBRegressor(**tabular_hyper)
        else:
            tabular_hyper = {
                "n_estimators": 150,
                "max_depth": 5,
                "seed": 0,
                "n_jobs": -1,
            }
            reg = XGBRegressor(**tabular_hyper)

    ## For both lightgbm and catboost, early stopping is specified in the parameters
    ## catboost will fail if there are extra parameters
    ## lightgbm will ignore extra parameters

    ### catboost
    if tabular_model == "catboost":
        if tabular_hyper:
            reg = CatBoostRegressor(**tabular_hyper)
        else:
            tabular_hyper = {
                "n_estimators": 150,
                "max_depth": 5,
                "random_seed": 0,
                "n_jobs": -1,
            }
            reg = CatBoostRegressor(**tabular_hyper)

    #### Fit Regressor Model for different ML methods
    if tabular_model in [
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
        "lightgbm-rf",
    ]:
        if extracted_features_test is not None:
            reg.fit(
                extracted_features_train,
                y_train,
                sample_weight=weights_train,
                eval_set=[(extracted_features_test, y_test)],
            )
        else:
            reg.fit(
                extracted_features_train,
                y_train,
            )

        if extracted_features_test is not None:
            valid_iteration = min(
                additional_hyper.get("gbm_start_iteration", 0),
                int(reg.booster_.num_trees() // 2),
            )
            if tabular_hyper.get("objective", "regression") == "cross_entropy":
                pred = reg.booster_.predict(
                    extracted_features_test, start_iteration=valid_iteration
                )
            else:
                pred = reg.predict(
                    extracted_features_test, start_iteration=valid_iteration
                )
            return reg, pred
        else:
            return reg

    ## xgboost ignores extra parameters
    if tabular_model in [
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        ## Ensemble Regressor
        if extracted_features_test is not None:
            ## xgboost in version 1.6.0 will have early_stopping moved to initlisation
            reg.fit(
                extracted_features_train,
                y_train,
                eval_set=[(extracted_features_test, y_test)],
                early_stopping_rounds=tabular_hyper["early_stopping_rounds"],
            )
        else:
            reg.fit(
                extracted_features_train,
                y_train,
            )

        if extracted_features_test is not None:
            valid_iteration = min(
                additional_hyper.get("gbm_start_iteration", 0),
                int(reg.get_booster().best_iteration // 2),
            )
            end_iteration = reg.get_booster().best_iteration
            pred = reg.predict(
                extracted_features_test,
                iteration_range=(valid_iteration, end_iteration),
            )
            return reg, pred
        else:
            return reg

    #### Fit Regressor Model for different ML methods
    if tabular_model in [
        "catboost",
    ]:
        if extracted_features_test is not None:
            reg.fit(
                extracted_features_train,
                y_train,
                sample_weight=weights_train,
                eval_set=[(extracted_features_test, y_test)],
            )
        else:
            reg.fit(
                extracted_features_train,
                y_train,
            )

        if extracted_features_test is not None:
            pred = reg.predict(extracted_features_test)
            return reg, pred
        else:
            return reg


def benchmark_neural_model(
    extracted_features_train,
    y_train,
    weights_train,
    extracted_features_test=None,
    y_test=None,
    weights_test=None,
    tabular_model="tabnet",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
):

    if tabular_model == "tabnet":
        if tabular_hyper:
            ## Default is PyTorch Adam Optimizer
            from torch.optim import Adam
            from torch.optim.lr_scheduler import StepLR

            tabnet_hyper = dict()
            tabnet_hyper["optimizer_fn"] = Adam
            tabnet_hyper["optimizer_params"] = {
                "lr": 0.02,
            }
            tabnet_hyper["scheduler_fn"] = StepLR
            tabnet_hyper["scheduler_params"] = {"gamma": 0.95, "step_size": 20}

            for key in [
                "seed",
                "n_d",
                "n_a",
                "n_steps",
                "n_independent",
                "n_shared",
                "gamma",
            ]:
                tabnet_hyper[key] = tabular_hyper[key]

            ## Separate Hyper-parameters in the fit function
            tabnet_fit_hyper = dict()
            for key in [
                "max_epochs",
                "patience",
                "batch_size",
            ]:
                tabnet_fit_hyper[key] = tabular_hyper[key]

            reg = TabNetRegressor(**tabnet_hyper)
        else:
            tabnet_hyper = {
                "n_d": 8,
                "n_a": 8,
                "seed": 0,
            }
            tabnet_fit_hyper = {
                "max_epochs": 10,
                "patience": 10,
                "batch_size": 1024,
            }
            reg = TabNetRegressor(**tabnet_hyper)

    if tabular_model in [
        "pytorch-tabular-tabtransformer",
        "pytorch-tabular-node",
        "pytorch-tabular-categoryembedding",
    ]:

        ## Create Configurations

        ## Assume all Data are continuous (as the case for Numerai)
        data_config = {
            "target": list(y_train.columns),
            "continuous_cols": list(extracted_features_train.columns),
            "categorical_cols": list(),
            "encode_date_columns": False,
            "continuous_feature_transform": None,
            "normalize_continuous_features": False,
            "num_workers": 5,
            "pin_memory": False,
            "continuous_dim": len(list(extracted_features_train.columns)),
            "categorical_dim": 0,
        }

        trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=tabular_hyper.get("batch_size", 1024),
            max_epochs=tabular_hyper.get("max_epochs", 100),
            gpus=tabular_hyper.get("gpus", 1),
        )

        optimizer_config = OptimizerConfig()

        if tabular_hyper["model_type"] == "CategoryEmbedding":
            model_config = CategoryEmbeddingModelConfig(
                task="regression",
                target_range=[
                    (-0.5, 0.5) for i in range(len(list(y_train.columns)))
                ],  ## ranking problem
                batch_norm_continuous_input=False,
                layers=tabular_hyper.get("layers", "1024-256-64"),
                activation=tabular_hyper.get("activation", "LeakyReLU"),
                dropout=tabular_hyper.get("dropout", 0.5),
            )

        if tabular_hyper["model_type"] == "TabTransformer":
            model_config = TabTransformerConfig(
                task="regression",
                target_range=[
                    (-0.5, 0.5) for i in range(len(list(y_train.columns)))
                ],  ## ranking problem
                out_ff_layers=tabular_hyper.get("out_ff_layers", "1024-256-64"),
                out_ff_activation=tabular_hyper.get("out_ff_activation", "LeakyReLU"),
                out_ff_dropout=tabular_hyper.get("out_ff_dropout", 0.5),
                num_heads=tabular_hyper.get("num_heads", 2),
                num_attn_blocks=tabular_hyper.get("num_attn_blocks", 2),
                attn_dropout=tabular_hyper.get("attn_dropout", 0.1),
                add_norm_dropout=tabular_hyper.get("add_norm_dropout", 0.1),
            )

        ## Out of Memory on our GPU

        if tabular_hyper["model_type"] == "AutoInt":
            model_config = AutoIntConfig(
                task="regression",
                target_range=[
                    (-0.5, 0.5) for i in range(len(list(y_train.columns)))
                ],  ## ranking problem
                layers=tabular_hyper.get("layers", "1024-256-64"),
                activation=tabular_hyper.get("activation", "LeakyReLU"),
                dropout=tabular_hyper.get("dropout", 0.5),
                attn_embed_dim=tabular_hyper.get("attn_embed_dim", 32),
                num_heads=tabular_hyper.get("num_heads", 2),
                num_attn_blocks=tabular_hyper.get("num_attn_blocks", 2),
                attn_dropouts=tabular_hyper.get("attn_dropouts", 0),
                embedding_dim=tabular_hyper.get("embedding_dim", 64),
                embedding_dropout=tabular_hyper.get("embedding_dropout", 0),
            )

        if tabular_hyper["model_type"] == "Node":
            model_config = NodeConfig(
                task="regression",
                target_range=[
                    (-0.5, 0.5) for i in range(len(list(y_train.columns)))
                ],  ## ranking problem
                num_trees=tabular_hyper.get("num_trees", 100),
                depth=tabular_hyper.get("depth", 6),
                additional_tree_output_dim=tabular_hyper.get(
                    "additional_tree_output_dim", 3
                ),
            )

        reg = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

    if tabular_model in [
        "Numerai-MLP",
    ]:
        config = dict()
        for key in [
            "neurons",
            "dropout",
            "max_epochs",
            "patience",
            "batch_size",
            "seed",
        ]:
            config[key] = tabular_hyper.get(key)
        reg = NumeraiMLP(config=config)

    #### Fit Models

    if tabular_model in [
        "tabnet",
    ]:
        if extracted_features_test is not None:
            reg.fit(
                extracted_features_train.values,
                y_train.values,
                eval_set=[(extracted_features_test.values, y_test.values)],
                max_epochs=tabnet_fit_hyper.get("max_epochs", 100),
                patience=tabnet_fit_hyper.get("patience", 10),
                batch_size=tabnet_fit_hyper.get("batch_size", 2048),
                virtual_batch_size=int(tabnet_fit_hyper.get("batch_size") / 4),
                num_workers=25,
            )
        else:
            reg.fit(
                extracted_features_train.values,
                y_train.values,  ## Reshape pd Series to 2D
                max_epochs=tabnet_fit_hyper.get("max_epochs", 100),
                patience=tabnet_fit_hyper.get("patience", 10),
                batch_size=tabnet_fit_hyper.get("batch_size", 2048),
                virtual_batch_size=int(tabnet_fit_hyper.get("batch_size") / 4),
                num_workers=25,
            )

        if extracted_features_test is not None:
            pred = reg.predict(extracted_features_test.values)
            return reg, pred
        else:
            return reg

    if tabular_model in [
        "pytorch-tabular-tabtransformer",
        "pytorch-tabular-node",
        "pytorch-tabular-categoryembedding",
    ]:
        train = pd.concat([extracted_features_train, y_train], axis=1)
        if extracted_features_test is not None:
            val = pd.concat([extracted_features_test, y_test], axis=1)
            reg.fit(
                train=train,
                validation=val,
            )
        else:
            reg.fit(
                train=train,
            )

        if extracted_features_test is not None:
            pred = reg.predict(extracted_features_test)
            return reg, pred[f"{list(y_train.columns)[0]}_prediction"].values
        else:
            return reg

    if tabular_model in [
        "Numerai-MLP",
    ]:
        reg.train(extracted_features_train, y_train, extracted_features_test, y_test)
        pred = reg.predict(extracted_features_test)
        return reg, pred


### Run ML pipeline for temporal tabular data

### Split data by temporal_cross_validation scheme
###


def benchmark_pipeline(
    features,
    target,
    weights,
    groups,
    model_params=None,
    feature_eng="numerai",
    feature_eng_parameters=None,
    tabular_model="lightgbm-gbdt",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
    n_jobs=20,
):
    ## Make sure features and target has the same index and dropna rows
    features, target = align_features_target(features, target)

    if debug:
        print(features.shape, target.shape, groups.shape)

    if not model_params:
        model_params = {
            "valid_splits": 1,
            "test_size": 52,
            "max_train_size": 52,
            "gap": 52,
            "cross_validation": "GroupedTimeSeriesSplit",
        }

    ## Cross Validation split
    if model_params["cross_validation"] == "GroupedTimeSeriesSplit":
        tscv = GroupedTimeSeriesSplit(
            valid_splits=model_params["valid_splits"],
            test_size=model_params["test_size"],
            max_train_size=model_params["max_train_size"],
            gap=model_params["gap"],
            debug=debug,
        )
    elif model_params["cross_validation"] == "GroupShuffleSplit":
        tscv = GroupShuffleSplit(
            n_splits=model_params["n_splits"],
            test_size=model_params["test_size"],
            train_size=model_params["train_size"],
            random_state=model_params.get("random_state", 0),
        )
    else:
        tscv = KFold(
            n_splits=model_params["n_splits"],
            shuffle=True,
            random_state=model_params.get("random_state", 0),
        )
    model_no = 1
    model_performance = dict()
    trained_models = dict()
    data = dict()
    parameters = dict()

    for train_index, test_index in tscv.split(features, groups=groups):

        model_name = "{}_{}_{}".format(feature_eng, tabular_model, model_no)
        ## Get Trained and Test Data
        try:
            X_train, X_test = features.loc[train_index, :], features.loc[test_index, :]
            y_train, y_test = target.loc[train_index, :], target.loc[test_index, :]
            ## Data Weights are pd Series
            weights_train, weights_test = (
                weights.loc[train_index],
                weights.loc[test_index],
            )
            ## Group Labels are pd Series
            group_train, group_test = (
                groups.loc[train_index],
                groups.loc[test_index],
            )
        except:
            ## KFold
            X_train, X_test = (
                features.iloc[train_index, :],
                features.iloc[test_index, :],
            )
            y_train, y_test = target.iloc[train_index, :], target.iloc[test_index, :]
            weights_train, weights_test = (
                weights.iloc[train_index],
                weights.iloc[test_index],
            )
            group_train, group_test = (
                groups.iloc[train_index],
                groups.iloc[test_index],
            )

        ##
        model_params["feature_columns"] = features.columns
        model_params["target_columns"] = target.columns

        parameters[model_name] = {
            "feature_eng": feature_eng_parameters.copy(),
            "tabular": tabular_hyper.copy(),
            "additional": additional_hyper.copy(),
        }

        if debug:
            print(
                "Train Size ",
                X_train.shape,
                "Test Size ",
                X_test.shape,
                "Prediction Size",
                y_train.shape,
            )

        ### Transform features

        (
            transformer,
            extracted_features_train,
            extracted_features_test,
        ) = benchmark_features_transform(
            X_train,
            y_train,
            X_test,
            group_train,
            group_test,
            feature_eng,
            feature_eng_parameters,
            n_jobs,
            debug,
        )

        model_params["feature_engineering"] = feature_eng
        model_params["tabular_model"] = tabular_model

        extracted_features_train, y_train = align_features_target(
            pd.DataFrame(extracted_features_train, index=X_train.index), y_train
        )
        extracted_features_test, y_test = align_features_target(
            pd.DataFrame(extracted_features_test, index=X_test.index), y_test
        )

        if debug:
            print("After feature eng transforms")
            print(
                "Train",
                extracted_features_train.shape,
                y_train.shape,
                extracted_features_train.columns,
                "Test",
                extracted_features_test.shape,
                y_test.shape,
                extracted_features_test.columns,
            )

        if tabular_model in [
            "lightgbm-gbdt",
            "lightgbm-goss",
            "lightgbm-dart",
            "lightgbm-rf",
            "xgboost-dart",
            "xgboost-gbtree",
            "catboost",
        ]:

            ### Train Tabular Models
            reg, pred = benchmark_tree_model(
                extracted_features_train,
                y_train,
                weights_train,
                extracted_features_test,
                y_test,
                weights_test,
                tabular_model,
                tabular_hyper,
                additional_hyper,
                debug,
            )

        elif tabular_model in [
            "tabnet",
            "pytorch-tabular-tabtransformer",
            "pytorch-tabular-node",
            "pytorch-tabular-categoryembedding",
            "Numerai-MLP",
        ]:

            ### Train Tabular Models
            reg, pred = benchmark_neural_model(
                extracted_features_train,
                y_train,
                weights_train,
                extracted_features_test,
                y_test,
                weights_test,
                tabular_model,
                tabular_hyper,
                additional_hyper,
                debug,
            )
        else:
            raise ValueError(f"Model {tabular_model} not supported")

        ## Convert Prediction output to a dataframe
        pred = pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)

        ### Compute model performance
        model_metrics = dict()
        model_metrics["MSE"] = mean_squared_error(y_test, pred)
        selected_correlations = list()
        for i in range(pred.shape[1]):
            selected_correlations.append(
                np.corrcoef(y_test.iloc[:, i], pred.iloc[:, i])[0, 1]
            )
        model_metrics["correlation"] = pd.Series(
            selected_correlations, index=y_test.columns
        )

        #### Training Dates
        model_params["train_start"] = group_train.iloc[0]
        model_params["train_end"] = group_train.iloc[-1]
        model_params["validation_start"] = group_test.iloc[0]
        model_params["validation_end"] = group_test.iloc[-1]
        parameters[model_name]["model"] = model_params.copy()

        ### Save model performance
        model_performance[model_name] = model_metrics.copy()

        trained_models[model_name] = {
            "transformer": transformer,
            "model": reg,
        }
        data[model_name] = {
            "X_train": extracted_features_train,
            "y_train": y_train,
            "X_validation": extracted_features_test,
            "y_validation": y_test,
            "prediction": pred,
        }

        model_no += 1
        if debug:
            print(
                model_performance[model_name],
                model_params,
                y_test.head(),
                pred.head(),
            )

        ## Clean up temp files produced by pytorch-tabular
        if tabular_model == "pytorch-tabular":
            shutil.rmtree(os.getcwd() + "/saved_models")

    return model_performance, trained_models, data, parameters
