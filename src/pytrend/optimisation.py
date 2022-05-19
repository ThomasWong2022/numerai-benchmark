#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Optimising hyper-parameters for ML models with hyperopt
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
import joblib, json, os, gc
import itertools

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

from .util import GroupedTimeSeriesSplit
from .benchmark import benchmark_pipeline, save_best_model, load_best_model
from .numerai import load_numerai_data, score_numerai


### Create Hyper-parameter space for hyperopt


def hyperopt_space(feature_eng="numerai", ml_method="lightgbm-gbdt"):

    space = {
        "proportion": hp.choice(
            "proportion",
            [0],
        ),
    }

    if feature_eng == "numerai":
        space["no_pca_features"] = hp.choice(
            "no_pca_features",
            [5 * i for i in range(0, 1)],
        )
        space["no_product_features"] = hp.choice(
            "no_product_features",
            [20 * i for i in range(0, 101)],
        )
        space["usesquare"] = hp.choice(
            "usesquare",
            [
                False,
            ],
        )
        space["dropout_pct"] = hp.choice(
            "dropout_pct",
            [0.01 * i for i in range(0, 11)],
        )

    if feature_eng == "numerai-pca":
        space["no_pca_features"] = hp.choice(
            "no_pca_features",
            [5 * i for i in range(0, 21)],
        )
        space["no_product_features"] = hp.choice(
            "no_product_features",
            [20 * i for i in range(0, 1)],
        )
        space["usesquare"] = hp.choice(
            "usesquare",
            [
                False,
            ],
        )
        space["dropout_pct"] = hp.choice(
            "dropout_pct",
            [0.01 * i for i in range(0, 11)],
        )

    if ml_method == "lightgbm-gbdt":
        space["n_estimators"] = hp.choice(
            "n_estimators",
            [32 * i for i in range(31, 129)],
        )
        space["learning_rate"] = hp.loguniform(
            "learning_rate",
            -6,
            -3,
        )
        space["max_depth"] = hp.choice(
            "max_depth",
            [i for i in range(1, 9)],
        )
        space["num_leaves"] = hp.choice(
            "num_leaves",
            [32 * i for i in range(1, 11)],
        )
        space["min_child_samples"] = hp.choice(
            "min_child_samples",
            [64 * i for i in range(1, 11)],
        )
        space["lambda_l1"] = hp.loguniform(
            "lambda_l1",
            -6,
            2,
        )
        space["lambda_l2"] = hp.loguniform(
            "lambda_l2",
            -6,
            2,
        )
        space["colsample_bytree"] = hp.choice(
            "colsample_bytree",
            [0.05 * i for i in range(2, 16)],
        )
        space["subsample"] = hp.choice("subsample", [0.1 * i for i in range(2, 9)])
        space["bagging_freq"] = hp.choice("bagging_freq", [5 * i for i in range(1, 6)])
        space["early_stopping_rounds"] = hp.choice(
            "early_stopping_rounds",
            [10 * i for i in range(1, 11)],
        )
        space["gbm_start_iteration"] = hp.choice(
            "gbm_start_iteration",
            [25 * i for i in range(0, 21)],
        )

    if ml_method == "lightgbm-dart":
        space["n_estimators"] = hp.choice(
            "n_estimators",
            [32 * i for i in range(31, 129)],
        )
        space["learning_rate"] = hp.loguniform(
            "learning_rate",
            -6,
            -3,
        )
        space["max_depth"] = hp.choice(
            "max_depth",
            [i for i in range(1, 9)],
        )
        space["num_leaves"] = hp.choice(
            "num_leaves",
            [32 * i for i in range(1, 11)],
        )
        space["min_child_samples"] = hp.choice(
            "min_child_samples",
            [64 * i for i in range(1, 11)],
        )
        space["lambda_l1"] = hp.loguniform(
            "lambda_l1",
            -6,
            2,
        )
        space["lambda_l2"] = hp.loguniform(
            "lambda_l2",
            -6,
            2,
        )
        space["colsample_bytree"] = hp.choice(
            "colsample_bytree",
            [0.05 * i for i in range(2, 16)],
        )
        space["subsample"] = hp.choice("subsample", [0.1 * i for i in range(2, 9)])
        space["bagging_freq"] = hp.choice("bagging_freq", [5 * i for i in range(1, 6)])
        space["drop_rate"] = hp.choice("drop_rate", [0.05 * i for i in range(1, 6)])
        space["skip_drop"] = hp.choice("skip_drop", [0.1 * i for i in range(5, 11)])
        space["early_stopping_rounds"] = hp.choice(
            "early_stopping_rounds",
            [10 * i for i in range(1, 11)],
        )
        space["gbm_start_iteration"] = hp.choice(
            "gbm_start_iteration",
            [25 * i for i in range(0, 21)],
        )
        
    if ml_method == "lightgbm-goss":
        space["n_estimators"] = hp.choice(
            "n_estimators",
            [32 * i for i in range(31, 129)],
        )
        space["learning_rate"] = hp.loguniform(
            "learning_rate",
            -6,
            -3,
        )
        space["max_depth"] = hp.choice(
            "max_depth",
            [i for i in range(1, 9)],
        )
        space["num_leaves"] = hp.choice(
            "num_leaves",
            [32 * i for i in range(1, 11)],
        )
        space["min_child_samples"] = hp.choice(
            "min_child_samples",
            [64 * i for i in range(1, 11)],
        )
        space["lambda_l1"] = hp.loguniform(
            "lambda_l1",
            -6,
            2,
        )
        space["lambda_l2"] = hp.loguniform(
            "lambda_l2",
            -6,
            2,
        )
        space["colsample_bytree"] = hp.choice(
            "colsample_bytree",
            [0.05 * i for i in range(2, 16)],
        )
        space["top_rate"] = hp.choice("top_rate", [0.1 * i for i in range(2, 7)])
        space["other_rate"] = hp.choice("other_rate", [0.05 * i for i in range(1, 7)])
        space["early_stopping_rounds"] = hp.choice(
            "early_stopping_rounds",
            [10 * i for i in range(1, 11)],
        )
        space["gbm_start_iteration"] = hp.choice(
            "gbm_start_iteration",
            [25 * i for i in range(0, 21)],
        )        
        
        

    if ml_method == "xgboost-gbtree":
        space["n_estimators"] = hp.choice(
            "n_estimators",
            [32 * i for i in range(31, 129)],
        )
        space["eta"] = hp.loguniform(
            "eta",
            -6,
            -3,
        )
        space["max_depth"] = hp.choice(
            "max_depth",
            [i for i in range(1, 9)],
        )
        space["num_leaves"] = hp.choice(
            "num_leaves",
            [32 * i for i in range(1, 11)],
        )
        space["min_child_weight"] = hp.loguniform(
            "min_child_weight",
            3,
            7,
        )
        space["alpha"] = hp.loguniform(
            "alpha",
            -6,
            2,
        )
        space["lambda"] = hp.loguniform(
            "lambda",
            -6,
            2,
        )
        space["colsample_bytree"] = hp.choice(
            "colsample_bytree",
            [0.05 * i for i in range(2, 16)],
        )
        space["subsample"] = hp.choice("subsample", [0.1 * i for i in range(2, 9)])
        space["early_stopping_rounds"] = hp.choice(
            "early_stopping_rounds",
            [10 * i for i in range(1, 11)],
        )
        space["gbm_start_iteration"] = hp.choice(
            "gbm_start_iteration",
            [25 * i for i in range(0, 21)],
        )

    if ml_method == "pytorch-tabular-tabtransformer":
        space["batch_size"] = hp.choice(
            "batch_size",
            [512 * i for i in range(1, 15)],
        )
        space["max_epochs"] = hp.choice(
            "max_epochs",
            [25 * i for i in range(1, 25)],
        )
        space["patience"] = hp.choice(
            "patience",
            [10 * i for i in range(1, 6)],
        )
        space["model_type"] = hp.choice(
            "model_type",
            [
                "TabTransformer",
            ],
        )
        space["out_ff_layers"] = hp.choice(
            "out_ff_layers",
            [
                "4096-1024-256",
                "2048-512-128",
                "1024-256-64",
                "4096-4096-4096",
                "2048-2048-2048",
                "1024-1024-1024",
                "256-256-256",
                "64-64-64",
                "4096-4096",
                "2048-2048",
                "1024-1024",
                "256-256",
                "64-64",
            ],
        )
        space["num_heads"] = hp.choice(
            "num_heads",
            [i for i in range(1, 2)],
        )
        space["num_attn_blocks"] = hp.choice(
            "num_attn_blocks",
            [i for i in range(2, 6)],
        )

    if ml_method == "pytorch-tabular-categoryembedding":
        space["batch_size"] = hp.choice(
            "batch_size",
            [512 * i for i in range(1, 15)],
        )
        space["max_epochs"] = hp.choice(
            "max_epochs",
            [25 * i for i in range(1, 25)],
        )
        space["patience"] = hp.choice(
            "patience",
            [10 * i for i in range(1, 6)],
        )
        space["model_type"] = hp.choice(
            "model_type",
            [
                "CategoryEmbedding",
            ],
        )
        space["layers"] = hp.choice(
            "layers",
            [
                "4096-1024-256",
                "2048-512-128",
                "1024-256-64",
                "4096-4096-4096",
                "2048-2048-2048",
                "1024-1024-1024",
                "256-256-256",
                "64-64-64",
                "4096-4096",
                "2048-2048",
                "1024-1024",
                "256-256",
                "64-64",
            ],
        )
        space["dropout"] = hp.choice(
            "dropout",
            [i * 0.1 for i in range(0, 8)],
        )

    if ml_method == "pytorch-tabular-node":
        space["batch_size"] = hp.choice(
            "batch_size",
            [512 * i for i in range(1, 15)],
        )
        space["max_epochs"] = hp.choice(
            "max_epochs",
            [25 * i for i in range(1, 25)],
        )
        space["patience"] = hp.choice(
            "patience",
            [10 * i for i in range(1, 6)],
        )
        space["model_type"] = hp.choice(
            "model_type",
            [
                "Node",
            ],
        )
        space["num_trees"] = hp.choice(
            "num_trees",
            [250 * i for i in range(1, 11)],
        )
        space["depth"] = hp.choice(
            "depth",
            [i for i in range(1, 3)],
        )

    if ml_method == "tabnet":
        space["batch_size"] = hp.choice(
            "batch_size",
            [512 * i for i in range(1, 15)],
        )
        space["max_epochs"] = hp.choice(
            "max_epochs",
            [25 * i for i in range(1, 25)],
        )
        space["patience"] = hp.choice(
            "patience",
            [10 * i for i in range(1, 6)],
        )
        space["n_d"] = hp.choice(
            "n_d",
            [4 * i for i in range(1, 5)],
        )
        space["n_a"] = hp.choice(
            "n_a",
            [4 * i for i in range(1, 5)],
        )
        space["n_steps"] = hp.choice(
            "n_steps",
            [i for i in range(2, 5)],
        )
        space["gamma"] = hp.choice(
            "gamma",
            [0.1 * i for i in range(10, 21)],
        )
        space["n_independent"] = hp.choice(
            "n_independent",
            [i for i in range(1, 3)],
        )
        space["n_shared"] = hp.choice(
            "n_shared",
            [i for i in range(1, 3)],
        )
        
    if ml_method == "Numerai-MLP":
        space["batch_size"] = hp.choice(
            "batch_size",
            [4096 * i for i in range(1, 5)],
        )
        space["max_epochs"] = hp.choice(
            "max_epochs",
            [10 * i for i in range(1, 25)],
        )
        space["patience"] = hp.choice(
            "patience",
            [5 * i for i in range(1, 6)],
        )
        space["dropout"] = hp.choice(
            "dropout",
            [i * 0.1 for i in range(0, 8)],
        )
        ## MLP layers combos 
        neuron_combos = list()
        for x in range(2,5):
            neuron_combos.extend(list(itertools.product(*[[256*2**i for i in range(0,4)] for g in range(x)])))
        space["neurons"] = hp.choice(
            "neurons",
            neuron_combos,
        )        

    return space


def create_model_parameters(
    args, feature_eng="numerai", ml_method="lightgbm-gbdt", seed=0, gpu_device_id=0,
):

    ## Check GPU-supported
    import subprocess

    try:
        subprocess.check_output("nvidia-smi")
        GPU_enabled = True
    except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
        GPU_enabled = False

    ## Feature Engineering
    if feature_eng == "numerai":
        feature_eng_parameters = {
            "usesquare": args["usesquare"],
            "no_product_features": args["no_product_features"],
            "no_pca_features": args["no_pca_features"],
            "dropout_pct": args["dropout_pct"],
            "seed": seed,
        }
    else:
        feature_eng_parameters = dict()

    ## xgboost
    if ml_method == "xgboost-gbtree":
        tabular_hyper = {
            "seed": seed,
            "verbosity": 0,
            "booster": "gbtree",
        }

        for key in [
            "n_estimators",
            "max_depth",
            "num_leaves",
            "min_child_weight",
            "alpha",
            "lambda",
            "colsample_bytree",
            "subsample",
            "eta",
            "early_stopping_rounds",
        ]:
            tabular_hyper[key] = args[key]

    ## lightgbm-gbdt
    if ml_method == "lightgbm-gbdt":
        tabular_hyper = {
            "seed": seed,
            "n_jobs": -1,
            "verbose": 0,
            "boosting": "gbdt",
        }

        if GPU_enabled:
            tabular_hyper["device"] = "gpu"
            tabular_hyper["gpu_device_id"] = gpu_device_id

        for key in [
            "n_estimators",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "lambda_l1",
            "lambda_l2",
            "colsample_bytree",
            "subsample",
            "bagging_freq",
            "learning_rate",
            "early_stopping_rounds",
        ]:
            tabular_hyper[key] = args[key]

    ## lightgbm-dart
    if ml_method == "lightgbm-dart":
        tabular_hyper = {
            "seed": seed,
            "n_jobs": -1,
            "verbose": 0,
            "boosting": "dart",
        }

        if GPU_enabled:
            tabular_hyper["device"] = "gpu"
            tabular_hyper["gpu_device_id"] = gpu_device_id

        for key in [
            "n_estimators",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "lambda_l1",
            "lambda_l2",
            "colsample_bytree",
            "subsample",
            "bagging_freq",
            "learning_rate",
            "early_stopping_rounds",
            "drop_rate",
            "skip_drop",
        ]:
            tabular_hyper[key] = args[key]
            
            
    ## lightgbm-dart
    if ml_method == "lightgbm-goss":
        tabular_hyper = {
            "seed": seed,
            "n_jobs": -1,
            "verbose": 0,
            "boosting": "goss",
        }

        if GPU_enabled:
            tabular_hyper["device"] = "gpu"
            tabular_hyper["gpu_device_id"] = gpu_device_id

        for key in [
            "n_estimators",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "lambda_l1",
            "lambda_l2",
            "colsample_bytree",
            "learning_rate",
            "early_stopping_rounds",
            "top_rate",
            "other_rate",
        ]:
            tabular_hyper[key] = args[key]            
            

    ## pytorch-tabular
    if ml_method == "pytorch-tabular-tabtransformer":
        tabular_hyper = {
            "seed": seed,
        }

        for key in [
            "batch_size",
            "max_epochs",
            "patience",
            "model_type",
            "out_ff_layers",
            "num_heads",
            "num_attn_blocks",
        ]:
            tabular_hyper[key] = args[key]

    ## pytorch-tabular
    if ml_method == "pytorch-tabular-categoryembedding":
        tabular_hyper = {
            "seed": seed,
        }

        for key in [
            "batch_size",
            "max_epochs",
            "patience",
            "model_type",
            "layers",
            "dropout",
        ]:
            tabular_hyper[key] = args[key]

    ## pytorch-tabular
    if ml_method == "pytorch-tabular-node":
        tabular_hyper = {
            "seed": seed,
        }

        for key in [
            "batch_size",
            "max_epochs",
            "patience",
            "model_type",
            "num_trees",
            "depth",
        ]:
            tabular_hyper[key] = args[key]

    if ml_method == "tabnet":
        tabular_hyper = {
            "seed": seed,
        }

        for key in [
            "batch_size",
            "max_epochs",
            "patience",
            "n_d",
            "n_a",
            "n_steps",
            "gamma",
            "n_independent",
            "n_shared",
        ]:
            tabular_hyper[key] = args[key]
            
    if ml_method == "Numerai-MLP":
        tabular_hyper = {
            "seed": seed,
        }

        for key in [
            "batch_size",
            "max_epochs",
            "patience",
            "dropout",
            "neurons",
        ]:
            tabular_hyper[key] = args[key]           
            
            

    ### Additional Hyper-parameters in .fit and .prediction
    if ml_method in [
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
        "lightgbm-rf",
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        additional_hyper = {
            "gbm_start_iteration": args["gbm_start_iteration"],
        }
    else:
        additional_hyper = dict()

    return feature_eng_parameters, tabular_hyper, additional_hyper


## Search for optimal hyper-parameter using hyperopt


def hyperopt_search(
    features,
    targets,
    groups,
    weights,
    seed=0,
    feature_eng="numerai",
    ml_method="lightgbm-gbdt",
    max_evals=10,
    model_params=None,
    gpu_device_id=0,
):

    if model_params is None:
        model_params = {
            "test_size": 52 * 2,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
        }

    # define an objective function
    def objective(args):

        (
            feature_eng_parameters,
            tabular_hyper,
            additional_hyper,
        ) = create_model_parameters(
            args, feature_eng=feature_eng, ml_method=ml_method, seed=seed, gpu_device_id=gpu_device_id,
        )

        print(
            ml_method,
            feature_eng_parameters,
            tabular_hyper,
            model_params,
            additional_hyper,
        )

        model_performance, trained_models, data, parameters = benchmark_pipeline(
            features,
            targets,
            weights,
            groups,
            feature_eng=feature_eng,
            feature_eng_parameters=feature_eng_parameters,
            tabular_model=ml_method,
            tabular_hyper=tabular_hyper,
            model_params=model_params,
            additional_hyper=additional_hyper,
            debug=False,
        )

        ## Get Predictions for each of the walk forward model
        predictions = list()
        for model_name in list(data.keys()):
            predictions.append(data[model_name]["prediction"])
        train_prediction_df = pd.DataFrame(pd.concat(predictions, axis=0).mean(axis=1))
        train_prediction_df.columns = ["prediction"]
        train_prediction_df["target"] = targets.reindex(train_prediction_df.index)
        train_prediction_df["era"] = groups.reindex(train_prediction_df.index)

        ## Score on Validation data
        train_prediction_df, correlations_by_era = score_numerai(
            train_prediction_df,
            features,
            riskiest_features=features.columns,
            proportion=args["proportion"],
            era_col="era",
            target_col_name="target",
        )

        if args["proportion"] > 0:
            sharpe = (
                correlations_by_era["neutralised_correlation"].mean()
                / correlations_by_era["neutralised_correlation"].std()
            )
        else:
            sharpe = (
                correlations_by_era["correlation"].mean()
                / correlations_by_era["correlation"].std()
            )

        print(f"Out of Sample Correlation {sharpe} Proportion {args['proportion']}")

        return -1 * sharpe

    ## Run Hyperopt
    space = hyperopt_space(feature_eng=feature_eng, ml_method=ml_method)
    trials = Trials()
    rng = np.random.default_rng(seed)
    best = fmin(
        objective,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=rng,
    )

    return space_eval(space, best), trials


def train_best_model(
    best_parameters,
    features,
    targets,
    groups,
    weights,
    seed=0,
    feature_eng="numerai",
    ml_method="lightgbm-gbdt",
    output_folder="numerai_models",
    model_params=None,
    gpu_device_id=0,
):

    if not os.path.exists(f"{output_folder}/"):
        os.mkdir(f"{output_folder}/")

    feature_eng_parameters, tabular_hyper, additional_hyper = create_model_parameters(
        best_parameters, feature_eng=feature_eng, ml_method=ml_method, seed=seed, gpu_device_id=gpu_device_id,
    )

    if model_params is None:
        model_params = {
            "test_size": 52 * 2,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
        }

    (model_performance, trained_models, data, parameters,) = benchmark_pipeline(
        features,
        targets,
        weights,
        groups,
        feature_eng=feature_eng,
        feature_eng_parameters=feature_eng_parameters,
        tabular_model=ml_method,
        tabular_hyper=tabular_hyper,
        model_params=model_params,
        additional_hyper=additional_hyper,
        debug=False,
    )

    if not ml_method in ["pytorch-tabular","pytorch-tabular-categoryembedding",]:
        output_model_path = f"{output_folder}/{ml_method}_seed{seed}.model"
    else:
        output_model_path = f"{output_folder}/{ml_method}_seed{seed}"

    output_parameters_path = f"{output_folder}/{ml_method}_seed{seed}.parameters"

    ## Currently We save the last Model only
    model_name = list(trained_models.keys())[-1]
    output_parameters = dict()
    output_parameters["parameters"] = parameters[model_name]
    output_parameters["transformer"] = trained_models[model_name]["transformer"]

    ## Save Parameters
    joblib.dump(output_parameters, output_parameters_path)

    ## Save Model
    save_best_model(
        trained_models[model_name]["model"],
        parameters[model_name]["model"]["tabular_model"],
        output_model_path,
    )

    return trained_models[model_name]["model"], output_parameters
