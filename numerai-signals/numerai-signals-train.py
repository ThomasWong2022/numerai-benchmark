import numpy as np
import pandas as pd
import os, json, datetime, sys
import lightgbm
import torch

from datetime import datetime

from pytrend.numerai import (
    run_numerai_models_performances,
)
from pytrend.optimisation import numerai_optimisation_pipeline_optuna

### Hyper-Parameter Space

optuna_xgboost_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "xgboost",
        "parameters": {
            "num_boost_round": ["int", {"low": 500, "high": 2000, "step": 50}],
            "eta": ["float", {"low": 0.001, "high": 0.1, "log": True}],
            "gamma": ["float", {"low": 0.0001, "high": 0.01, "log": True}],
            "max_depth": ["int", {"low": 4, "high": 7, "step": 1}],
            "subsample": ["float", {"low": 0.5, "high": 1, "step": 0.05}],
            "colsample_bytree": ["float", {"low": 0.05, "high": 0.25, "step": 0.01}],
            "alpha": ["float", {"low": 0.01, "high": 1, "log": True}],
            "lambda": ["float", {"low": 0.01, "high": 1, "log": True}],
            "rate_drop": ["float", {"low": 0.05, "high": 0.5, "step": 0.05}],
            "skip_drop": ["float", {"low": 0.05, "high": 0.5, "step": 0.05}],
            "early_stopping_rounds": ["int", {"low": 5000, "high": 6000, "step": 500}],
            "booster": [
                "categorical",
                {
                    "choices": [
                        "gbtree",
                        "dart",
                    ]
                },
            ],
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist",
            "verbosity": 1,
            "max_bin": 15,
        },
    },
    "model_params": {
        "train": {
            "test_size": 52 * 4,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 52 * 4,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_20d"],
        "train_endera": datetime.strptime("2015-12-31", "%Y-%m-%d"),
        "validate_targets": [
            "target_20d",
            "target_20d_raw_return",
            "target_20d_factor_neutral",
            "target_20d_factor_feat_neutral",
            "target_4d",
        ],
        "validate_enderas": [
            datetime.strptime("2015-12-31", "%Y-%m-%d"),
            datetime.strptime("2016-12-31", "%Y-%m-%d"),
            datetime.strptime("2017-12-31", "%Y-%m-%d"),
            datetime.strptime("2018-12-31", "%Y-%m-%d"),
            datetime.strptime("2019-12-31", "%Y-%m-%d"),
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai-signals-models",
        "model_no_start": None,
        "no_models_per_config": 20,
        "feature_sets": "signals",
    },
}


optuna_lightgbm_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "lightgbm",
        "parameters": {
            "num_iterations": ["int", {"low": 50, "high": 1000, "step": 50}],
            "learning_rate": ["float", {"low": 0.005, "high": 0.1, "log": True}],
            "min_data_in_leaf": ["int", {"low": 2500, "high": 40000, "step": 2500}],
            "feature_fraction": [
                "float",
                {"low": 0.1, "high": 1, "step": 0.05},
            ],
            "lambda_l1": ["float", {"low": 0.01, "high": 1, "log": True}],
            "lambda_l2": ["float", {"low": 0.01, "high": 1, "log": True}],
            "bagging_fraction": ["float", {"low": 0.5, "high": 1, "step": 0.05}],
            "bagging_freq": ["int", {"low": 10, "high": 50, "step": 5}],
            "drop_rate": ["float", {"low": 0.1, "high": 0.3, "step": 0.05}],
            "skip_drop": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
            "top_rate": ["float", {"low": 0.1, "high": 0.4, "step": 0.05}],
            "other_rate": ["float", {"low": 0.05, "high": 0.2, "step": 0.05}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "gbdt",
                    ]
                },
            ],
            "early_stopping_round": ["int", {"low": 5000, "high": 6000, "step": 500}],
            "objective": "regression",
            "device_type": "gpu",
            "num_threads": 0,
            "verbosity": -1,
            "num_gpu": 1,
            "max_bin": 7,
            "gpu_use_dp": False,
        },
    },
    "model_params": {
        "train": {
            "test_size": 52 * 4,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 52 * 4,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 52,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_20d"],
        "train_endera": datetime.strptime("2015-12-31", "%Y-%m-%d"),
        "validate_targets": [
            "target_20d",
            "raw_return_target_20d",
            "factor_neutral_target_20d",
            "factor_feat_neutral_target_20d",
            "target_4d",
        ],
        "validate_enderas": [
            datetime.strptime("2015-12-31", "%Y-%m-%d"),
            datetime.strptime("2016-12-31", "%Y-%m-%d"),
            datetime.strptime("2017-12-31", "%Y-%m-%d"),
            datetime.strptime("2018-12-31", "%Y-%m-%d"),
            datetime.strptime("2019-12-31", "%Y-%m-%d"),
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai-signals-models",
        "model_no_start": None,
        "no_models_per_config": 20,
        "feature_sets": "signals",
        "mix_cv": True,
        "mix_targets": False,
    },
}

model_no_start = 6000

feature_sets = [
    "all",
    "signature",
    "catch22",
    "stats",
    "financials",
    "ravenpack",
]

feature_sets = [
    "price",
]

for feature_set in feature_sets:

    numerai_files = {
        "dataset": f"signals-data/numerai_signals_features_{feature_set}.parquet",
        "feature_metadata": f"signals-data/numerai_signals_features_{feature_set}_metadata.json",
    }
    optimisation_args = optuna_lightgbm_args
    optimisation_args["model_params"]["model_no_start"] = model_no_start

    MODEL_FOLDER = "numerai-signals-models"
    PERFORMANCES_FOLDER = "numerai-signals-performances"
    if not os.path.exists(f"{MODEL_FOLDER}/"):
        os.mkdir(f"{MODEL_FOLDER}/")
    if not os.path.exists(f"{PERFORMANCES_FOLDER}/"):
        os.mkdir(f"{PERFORMANCES_FOLDER}/")

    ### Logging
    log_file_path = f'{optimisation_args["ml_method"]["method"]}_{optimisation_args["model_params"]["model_no_start"]}.log'
    import logging

    logger = logging.getLogger("Numerai")
    logger.setLevel(level=logging.DEBUG)
    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename=log_file_path)
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(level=logging.DEBUG)
    logger.addHandler(fileHandler)

    def defaultconverter(o):
        if isinstance(o, datetime):
            return o.__str__()

    #### Save Search Space
    search_space_file_path = f'search_space_{optimisation_args["ml_method"]["method"]}_{optimisation_args["model_params"]["model_no_start"]}.json'
    if not os.path.exists(search_space_file_path):
        with open(search_space_file_path, "w") as f:
            json.dump(optimisation_args, f, default=defaultconverter)

    optimised_file_path = f'best_parameters_{optimisation_args["ml_method"]["method"]}_{optimisation_args["model_params"]["model_no_start"]}.json'
    if os.path.exists(optimised_file_path):
        run_optimisation = False
    else:
        run_optimisation = True

    ### Train ML models
    numerai_optimisation_pipeline_optuna(
        optimisation_args,
        numerai_files=numerai_files,
        run_optimisation=run_optimisation,
        optimised_parameters_path=optimised_file_path,
        grid_search_seed=0,
        n_trials=100,
        timeout=3600 * 8,
        debug=False,
    )

    start = optimisation_args["model_params"]["model_no_start"]
    end = optimisation_args["model_params"]["model_no_start"] + optimisation_args[
        "model_params"
    ]["no_models_per_config"] * len(
        optimisation_args["model_params"]["validate_targets"]
    ) * len(
        optimisation_args["model_params"]["validate_enderas"]
    )
    for seed in range(
        start, end, optimisation_args["model_params"]["no_models_per_config"]
    ):
        Numerai_Model_Names = [
            f'{MODEL_FOLDER}/{optimisation_args["ml_method"]["method"]}_{optimisation_args["feature_eng"]["method"]}_1_{seed+seq}.parameters'
            for seq in range(optimisation_args["model_params"]["no_models_per_config"])
        ]
        ## Check if performanc already exists
        no_models = len(Numerai_Model_Names)
        stem = Numerai_Model_Names[0].split("/")[-1].replace(".parameters", "")
        correlations_filename = f"{PERFORMANCES_FOLDER}/{stem}_{no_models}.csv"
        if not os.path.exists(correlations_filename):
            run_numerai_models_performances(
                Numerai_Model_Names,
                None,
                None,
                PERFORMANCES_FOLDER,
                data_file=numerai_files["dataset"],
                data_version="signals",
                target_col=["target_20d"],
            )

    model_no_start = model_no_start + 1000
