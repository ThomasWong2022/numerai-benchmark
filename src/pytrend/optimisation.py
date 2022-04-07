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

from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

from .benchmark import benchmark_pipeline
from .util import save_best_model, load_best_model, GroupedTimeSeriesSplit
from .numerai import load_numerai_data, score_numerai


### Create Hyper-parameter space for hyperopt


def hyperopt_space(feature_eng="numerai", ml_method="lightgbm-gbdt"):

    space = {
        "proportion": hp.choice(
            "proportion",
            [0.25 * i for i in range(0, 5)],
        ),
    }

    if feature_eng == "numerai":
        space["no_pca_features"] = hp.choice(
            "no_pca_features",
            [5 * i for i in range(0, 5)],
        )        
        space["no_product_features"] = hp.choice(
            "no_product_features",
            [20 * i for i in range(0, 101)],
        )
        space["usesquare"] = hp.choice(
            "usesquare",
            [
                False,
                True,
            ],
        )
        space["dropout_pct"] = hp.choice(
            "dropout_pct",
            [0.01 * i for i in range(0, 11)],
        )

    if ml_method == "lightgbm-gbdt":
        space["n_estimators"] = hp.choice(
            "n_estimators",
            [64 * i for i in range(1, 41)],
        )
        space["learning_rate"] = hp.choice(
            "learning_rate",
            [0.001 * i for i in range(1, 6)],
        )
        space["max_depth"] = hp.choice(
            "max_depth",
            [i for i in range(3, 11)],
        )
        space["num_leaves"] = hp.choice(
            "num_leaves",
            [32 * i for i in range(1, 11)],
        )
        space["min_child_samples"] = hp.choice(
            "min_child_samples",
            [128 * i for i in range(1, 11)],
        )
        space["lambda_l1"] = hp.choice(
            "lambda_l1",
            [0.25 * i for i in range(0, 16)],
        )
        space["lambda_l2"] = hp.choice(
            "lambda_l2",
            [0.25 * i for i in range(0, 16)],
        )
        space["colsample_bytree"] = hp.choice(
            "colsample_bytree",
            [0.05 * i for i in range(1, 21)],
        )
        space["subsample"] = hp.choice("subsample", [0.25 * i for i in range(1, 5)])
        space["bagging_freq"] = hp.choice("bagging_freq", [5 * i for i in range(1, 5)])
        space["early_stopping_rounds"] = hp.choice(
            "early_stopping_rounds",
            [25 * i for i in range(1, 5)],
        )
        space["lgb_start_iteration"] = hp.choice(
            "lgb_start_iteration",
            [100 * i for i in range(0, 4)],
        )
    return space


def create_model_parameters(
    args, feature_eng="numerai", ml_method="lightgbm-gbdt", seed=0
):
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
        

    ## CatBoost
    if ml_method == "catboost":
        tabular_hyper = {
            "random_seed": seed,
            "verbose": False,
        }

        for key in [
            "n_estimators",
            "max_depth",
            "min_child_samples",
            "colsample_bylevel",
            "subsample",
            "learning_rate",
            "early_stopping_rounds",
        ]:
            tabular_hyper[key] = args[key]

    ## xgboost
    if ml_method == "xgboost":
        tabular_hyper = {
            "seed": seed,
            "verbosity": 0,
        }

        for key in [
            "n_estimators",
            "max_depth",
            "max_leaves",
            "min_child_weight",
            "colsample_bytree",
            "subsample",
            "learning_rate",
            "early_stopping_rounds",
        ]:
            tabular_hyper[key] = args[key]

    ## lightgbm
    if ml_method == "lightgbm-gbdt":
        tabular_hyper = {
            "seed": seed,
            "n_jobs": -1,
            "verbose": 0,
            "boosting": "gbdt",
        }

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
            
      ### Additional Hyper-parameters in .fit and .prediction 
    if ml_method in ["lightgbm-gbdt"]:
        additional_hyper = {'lgb_start_iteration': args['lgb_start_iteration'],}
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
):

    if model_params is None:
        model_params = {
            "test_size": 52 * 5,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
        }

    # define an objective function
    def objective(args):

        feature_eng_parameters, tabular_hyper, additional_hyper = create_model_parameters(
            args, feature_eng=feature_eng, ml_method=ml_method, seed=seed
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
):

    if not os.path.exists(f"{output_folder}/"):
        os.mkdir(f"{output_folder}/")

    feature_eng_parameters, tabular_hyper = create_model_parameters(
        best_parameters, feature_eng=feature_eng, ml_method=ml_method, seed=seed
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
        debug=False,
    )

    if not ml_method in ["pytorch_tabular"]:
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
