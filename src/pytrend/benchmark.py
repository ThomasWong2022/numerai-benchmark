#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of GBDT models for temporal tabular data
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
import joblib, os, shutil, datetime


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit

## Machine Learning packages
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor

import lightgbm, xgboost, catboost
import torch

import logging, gc

if torch.cuda.is_available():
    import cupy as cp


from .util import align_features_target, RollingTSTransformer, GroupedTimeSeriesSplit
from .feature import benchmark_features_transform
from .neural import TabularModel, MLP, LSTM_Tabular

# ## Persistence of ML models

### Save Best Model using method provided
def save_best_model(model, model_type, outputpath):
    if model_type in [
        "lightgbm",
        "lightgbm-gbdt",
        "lightgbm-goss",
        "lightgbm-rf",
        "lightgbm-dart",
    ]:
        model.save_model(outputpath)
    if model_type in [
        "xgboost",
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        model.save_model(outputpath)
    if model_type == "catboost":
        model.save_model(outputpath)
    if model_type in [
        "Numerai-MLP",
        "Numerai-LSTM",
    ]:
        model.save_model(outputpath)
    if model_type == "tabnet":
        model.save_model(outputpath)
        os.rename("{}.zip".format(outputpath), outputpath)
    return None


### load Best Model using method provided
def load_best_model(model_type, outputpath):
    if model_type in [
        "lightgbm",
        "lightgbm-gbdt",
        "lightgbm-goss",
        "lightgbm-rf",
        "lightgbm-dart",
    ]:
        reg = lightgbm.Booster(model_file=outputpath)
    if model_type in [
        "xgboost",
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        reg = xgboost.Booster()
        reg.load_model(outputpath)
    if model_type == "catboost":
        reg = catboost.CatBoost()
        reg.load_model(outputpath)
    if model_type in [
        "Numerai-MLP",
    ]:
        reg = TabularModel(MLP, config=dict())
        reg.load_model(outputpath)
    if model_type in [
        "Numerai-LSTM",
    ]:
        reg = TabularModel(LSTM_Tabular, config=dict())
        reg.load_model(outputpath)
    if model_type == "tabnet":
        from pytorch_tabnet.tab_model import TabNetRegressor

        reg = TabNetRegressor()
        reg.load_model(outputpath)
    if model_type == "feature-momentum":
        reg = None
    return reg


# ## Fit ML Models


def benchmark_neural_model(
    extracted_features_train,
    y_train,
    weights_train,
    extracted_features_test=None,
    y_test=None,
    weights_test=None,
    tabular_model="Numerai-MLP",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
):

    gc.collect()

    ## Initialise and Train Models
    if tabular_model in [
        "Numerai-MLP",
    ]:
        reg = TabularModel(MLP, config=tabular_hyper)
        reg.train(extracted_features_train, y_train, extracted_features_test, y_test)
        pred = reg.predict(extracted_features_test.values)
        return reg, pred

    if tabular_model in [
        "Numerai-LSTM",
    ]:
        reg = TabularModel(LSTM_Tabular, config=tabular_hyper)
        reg.train(extracted_features_train, y_train, extracted_features_test, y_test)
        pred = reg.predict(extracted_features_test.values)
        return reg, pred

    if tabular_model == "tabnet":
        ## Default is PyTorch Adam Optimizer
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR
        from pytorch_tabnet.tab_model import TabNetRegressor

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
            "momentum",
            "lambda_sparse",
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
        reg.fit(
            extracted_features_train.values,
            y_train.values,
            eval_set=[(extracted_features_test.values, y_test.values)],
            max_epochs=tabnet_fit_hyper.get("max_epochs", 20),
            patience=tabnet_fit_hyper.get("patience", 5),
            batch_size=tabnet_fit_hyper.get("batch_size", 40960),
            virtual_batch_size=int(tabnet_fit_hyper.get("batch_size", 40960) / 4),
            num_workers=0,
        )
        pred = reg.predict(extracted_features_test.values)
        return reg, pred


def benchmark_tree_model(
    extracted_features_train,
    y_train,
    weights_train,
    extracted_features_test=None,
    y_test=None,
    weights_test=None,
    tabular_model="lightgbm",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
):
    ### Free up Memory from previous loop
    gc.collect()

    #### Fit Regressor Model for different ML methods
    if tabular_model in [
        "lightgbm",
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
        "lightgbm-rf",
    ]:
        if y_test is not None:
            train_data = lightgbm.Dataset(
                extracted_features_train,
                label=y_train,
                weight=weights_train,
                params={"max_bin": tabular_hyper["max_bin"]},
            )
            test_data = lightgbm.Dataset(
                extracted_features_test,
                label=y_test,
                weight=weights_test,
                params={"max_bin": tabular_hyper["max_bin"]},
            )
            early_stopping_rounds = tabular_hyper.get("early_stopping_round", 0)
            model = lightgbm.train(
                tabular_hyper,
                train_set=train_data,
                num_boost_round=tabular_hyper["num_iterations"],
                valid_sets=[test_data],
                callbacks=[
                    lightgbm.log_evaluation(period=1000),
                    lightgbm.early_stopping(early_stopping_rounds),
                ],
            )
            valid_iteration = min(
                additional_hyper.get("gbm_start_iteration", 0),
                int(model.num_trees() // 2),
            )
            pred = model.predict(
                extracted_features_test, start_iteration=valid_iteration
            )
            return model, pred
        else:
            train_data = lightgbm.Dataset(
                extracted_features_train,
                label=y_train,
                weight=weights_train,
            )
            model = lightgbm.train(
                tabular_hyper,
                train_set=train_data,
                num_boost_round=tabular_hyper["num_iterations"],
            )
            return model

    ## xgboost ignores extra parameters
    if tabular_model in [
        "xgboost",
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        ## Create DMatrix
        if y_test is not None:
            train_data = xgboost.DMatrix(
                extracted_features_train,
                label=y_train.values.reshape(-1),
                weight=weights_train.values.reshape(-1),
            )
            test_data = xgboost.DMatrix(
                extracted_features_test,
                label=y_test.values.reshape(-1),
                weight=weights_test.values.reshape(-1),
            )
            ### Train XGBoost model
            model = xgboost.train(
                tabular_hyper,
                train_data,
                num_boost_round=tabular_hyper["num_boost_round"],
                evals=[(test_data, "xgboost_test_data")],
                early_stopping_rounds=tabular_hyper["early_stopping_rounds"],
                verbose_eval=100,
            )
            start_iteration = min(
                additional_hyper.get("gbm_start_iteration", 0),
                int(model.best_iteration // 2),
            )
            end_iteration = model.best_iteration
            pred = model.predict(
                test_data,
                iteration_range=(start_iteration, end_iteration),
            )
            return model, pred
        else:
            train_data = xgboost.DMatrix(
                extracted_features_train,
                label=y_train.values.reshape(-1),
                weight=weights_train.values.reshape(-1),
            )
            model = xgboost.train(
                tabular_hyper,
                train_data,
                num_boost_round=tabular_hyper["num_boost_round"],
            )
            return model


### Run ML pipeline for temporal tabular data
def benchmark_pipeline(
    features,
    target,
    weights,
    groups,
    model_params=None,
    feature_eng=None,
    feature_eng_parameters=None,
    tabular_model="lightgbm",
    tabular_hyper=None,
    additional_hyper=None,
    debug=False,
):

    if debug:
        print(f"Dataset Sizes {features.shape} {target.shape} {groups.shape}")

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

        ## Get Trained and Test Data
        if model_params["cross_validation"] == "GroupedTimeSeriesSplit":
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
            if debug:
                print(X_train.shape, X_test.shape)

        ## For Existing Cross Validation Splits in scikit-learn it is based on index location (iloc)
        else:
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
            debug,
        )

        if tabular_model in [
            "lightgbm-gbdt",
            "lightgbm-goss",
            "lightgbm-dart",
            "lightgbm-rf",
            "xgboost-dart",
            "xgboost-gbtree",
            "catboost",
            "lightgbm",
            "xgboost",
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

        if tabular_model in [
            "Numerai-MLP",
            "Numerai-LSTM",
            "tabnet",
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

        ## Convert Prediction output to a dataframe
        pred = pd.DataFrame(pred, index=y_test.index, columns=y_test.columns)

        model_name = "{}_{}_{}".format(tabular_model, feature_eng, model_no)

        parameters[model_name] = {
            "feature_eng": feature_eng_parameters.copy(),
            "tabular": tabular_hyper.copy(),
            "additional": additional_hyper.copy(),
        }

        ### Compute model performance
        model_metrics = dict()
        model_metrics["MSE"] = mean_squared_error(y_test, pred)
        model_performance[model_name] = model_metrics.copy()

        #### Training Parameters
        model_params["feature_columns"] = features.columns
        model_params["target_columns"] = target.columns
        model_params["feature_engineering"] = feature_eng
        model_params["tabular_model"] = tabular_model
        model_params["train_start"] = group_train.iloc[0]
        model_params["train_end"] = group_train.iloc[-1]
        model_params["validation_start"] = group_test.iloc[0]
        model_params["validation_end"] = group_test.iloc[-1]
        model_params["model_name"] = model_name
        parameters[model_name]["model"] = model_params.copy()

        if debug:
            print(parameters[model_name])

        if transformer is not None:
            trained_models[model_name] = {
                "transformer": transformer.data,
                "model": reg,
            }
        else:
            trained_models[model_name] = {
                "transformer": None,
                "model": reg,
            }

        data[model_name] = {
            "prediction": pred,
            "y_test": y_test,
        }

        model_no += 1

    return model_performance, trained_models, data, parameters
