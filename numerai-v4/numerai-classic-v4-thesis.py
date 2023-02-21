import numpy as np
import pandas as pd
import os, json, datetime, sys
import lightgbm
import torch
from pytrend.numerai import (
    run_numerai_models_performances,
)

from pytrend.optimisation import numerai_optimisation_pipeline_optuna


### Hyper-Parameter Space

optuna_mlp_simple_args_v2 = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "Numerai-MLP",
        "parameters": {
            "learning_rate": ["float", {"low": 0.00005, "high": 0.01, "log": True}],
            "max_epochs": ["int", {"low": 30, "high": 80, "step": 5}],
            "patience": ["int", {"low": 5, "high": 10, "step": 5}],
            "dropout": ["float", {"low": 0.05, "high": 0.5, "log": True}],
            "num_layers": ["int", {"low": 2, "high": 7, "step": 1}],
            "neurons": ["int", {"low": 128, "high": 1024, "step": 128}],
            "neuron_scale": ["float", {"low": 0.2, "high": 1, "log": True}],
            "batch_size": ["int", {"low": 10240, "high": 40960, "step": 10240}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_startera": "0001",
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 3000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
        "mix_cv": False,
        "mix_targets": False,
    },
}


optuna_mlp_args_v2 = {
    "feature_eng": {
        "method": "numerai",
        "parameters": {
            "no_product_features": ["int", {"low": 50, "high": 1000, "step": 50}],
            "dropout_pct": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
        },
    },
    "ml_method": {
        "method": "Numerai-MLP",
        "parameters": {
            "learning_rate": ["float", {"low": 0.00005, "high": 0.01, "log": True}],
            "max_epochs": ["int", {"low": 30, "high": 80, "step": 5}],
            "patience": ["int", {"low": 5, "high": 10, "step": 5}],
            "dropout": ["float", {"low": 0.05, "high": 0.5, "log": True}],
            "num_layers": ["int", {"low": 2, "high": 7, "step": 1}],
            "neurons": ["int", {"low": 128, "high": 1024, "step": 128}],
            "neuron_scale": ["float", {"low": 0.2, "high": 1, "log": True}],
            "batch_size": ["int", {"low": 10240, "high": 40960, "step": 10240}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_startera": "0001",
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 1000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
        "mix_cv": False,
        "mix_targets": False,
    },
}



optuna_mlp_simple_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "Numerai-MLP",
        "parameters": {
            "max_epochs": ["int", {"low": 10, "high": 100, "step": 5}],
            "patience": ["int", {"low": 5, "high": 20, "step": 5}],
            "dropout": ["float", {"low": 0.1, "high": 0.9, "log": True}],
            "num_layers": ["int", {"low": 2, "high": 7, "step": 1}],
            "neurons": ["int", {"low": 64, "high": 1024, "step": 64}],
            "neuron_scale": ["float", {"low": 0.3, "high": 1, "log": True}],
            "batch_size": ["int", {"low": 10240, "high": 40960, "step": 10240}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 2000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_mlp_args = {
    "feature_eng": {
        "method": "numerai",
        "parameters": {
            "no_product_features": ["int", {"low": 50, "high": 1000, "step": 50}],
            "dropout_pct": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
        },
    },
    "ml_method": {
        "method": "Numerai-MLP",
        "parameters": {
            "max_epochs": ["int", {"low": 10, "high": 100, "step": 5}],
            "patience": ["int", {"low": 5, "high": 20, "step": 5}],
            "dropout": ["float", {"low": 0.1, "high": 0.9, "log": True}],
            "num_layers": ["int", {"low": 2, "high": 7, "step": 1}],
            "neurons": ["int", {"low": 64, "high": 1024, "step": 64}],
            "neuron_scale": ["float", {"low": 0.3, "high": 1, "log": True}],
            "batch_size": ["int", {"low": 10240, "high": 40960, "step": 10240}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 0,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_tabnet_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "tabnet",
        "parameters": {
            "max_epochs": ["int", {"low": 10, "high": 50, "step": 5}],
            "patience": ["int", {"low": 5, "high": 20, "step": 5}],
            "batch_size": ["int", {"low": 1024, "high": 4096, "step": 1024}],
            "n_d": ["int", {"low": 4, "high": 16, "step": 4}],
            "n_a": ["int", {"low": 4, "high": 16, "step": 4}],
            "n_steps": ["int", {"low": 1, "high": 3, "step": 1}],
            "n_shared": ["int", {"low": 1, "high": 3, "step": 1}],
            "n_independent": ["int", {"low": 1, "high": 3, "step": 1}],
            "gamma": ["float", {"low": 1, "high": 2, "step": 0.1}],
            "momentum": ["float", {"low": 0.01, "high": 0.4, "step": 0.01}],
            "lambda_sparse": ["float", {"low": 0.0001, "high": 0.01, "log": True}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 20,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 4,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 20,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 4,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
        ],
        "train_resample_freq": 5,
        "validate_resample_freq": 5,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 2000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_tabnet_advanced_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "tabnet",
        "parameters": {
            "max_epochs": ["int", {"low": 10, "high": 50, "step": 5}],
            "patience": ["int", {"low": 5, "high": 20, "step": 5}],
            "batch_size": ["int", {"low": 1024, "high": 4096, "step": 1024}],
            "n_d": ["int", {"low": 4, "high": 64, "step": 4}],
            "n_a": ["int", {"low": 4, "high": 64, "step": 4}],
            "n_steps": ["int", {"low": 1, "high": 5, "step": 1}],
            "n_shared": ["int", {"low": 1, "high": 5, "step": 1}],
            "n_independent": ["int", {"low": 1, "high": 5, "step": 1}],
            "gamma": ["float", {"low": 1, "high": 2, "step": 0.1}],
            "momentum": ["float", {"low": 0.01, "high": 0.4, "step": 0.01}],
            "lambda_sparse": ["float", {"low": 0.0001, "high": 0.01, "log": True}],
        },
    },
    "model_params": {
        "train": {
            "test_size": 20,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 4,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 20,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 4,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
        ],
        "train_resample_freq": 5,
        "validate_resample_freq": 5,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 1000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


### Hyper-Parameter Space
optuna_lightgbm_gbdt_args = {
    "feature_eng": {
        "method": "numerai",
        "parameters": {
            "no_product_features": ["int", {"low": 50, "high": 1000, "step": 50}],
            "dropout_pct": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
        },
    },
    "ml_method": {
        "method": "lightgbm-gbdt",
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
            "early_stopping_round": ["int", {"low": 5000, "high": 6000, "step": 500}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "gbdt",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 0,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_lightgbm_gbdt_simple_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "lightgbm-gbdt",
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
            "early_stopping_round": ["int", {"low": 5000, "high": 6000, "step": 500}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "gbdt",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 2000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_lightgbm_dart_args = {
    "feature_eng": {
        "method": "numerai",
        "parameters": {
            "no_product_features": ["int", {"low": 50, "high": 1000, "step": 50}],
            "dropout_pct": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
        },
    },
    "ml_method": {
        "method": "lightgbm-dart",
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
            "drop_rate": ["float", {"low": 0.1, "high": 0.5, "step": 0.1}],
            "skip_drop": ["float", {"low": 0.1, "high": 0.8, "step": 0.1}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "dart",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 0,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_lightgbm_dart_simple_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "lightgbm-dart",
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
            "drop_rate": ["float", {"low": 0.1, "high": 0.5, "step": 0.1}],
            "skip_drop": ["float", {"low": 0.1, "high": 0.8, "step": 0.1}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "dart",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 2000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}

optuna_lightgbm_goss_args = {
    "feature_eng": {
        "method": "numerai",
        "parameters": {
            "no_product_features": ["int", {"low": 50, "high": 1000, "step": 50}],
            "dropout_pct": ["float", {"low": 0.05, "high": 0.25, "step": 0.05}],
        },
    },
    "ml_method": {
        "method": "lightgbm-goss",
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
            "top_rate": ["float", {"low": 0.1, "high": 0.4, "step": 0.05}],
            "other_rate": ["float", {"low": 0.05, "high": 0.2, "step": 0.05}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "goss",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 0,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


optuna_lightgbm_goss_simple_args = {
    "feature_eng": {
        "method": None,
        "parameters": {},
    },
    "ml_method": {
        "method": "lightgbm-goss",
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
            "top_rate": ["float", {"low": 0.1, "high": 0.4, "step": 0.05}],
            "other_rate": ["float", {"low": 0.05, "high": 0.2, "step": 0.05}],
            "boosting": [
                "categorical",
                {
                    "choices": [
                        "goss",
                    ]
                },
            ],
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
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "validate": {
            "test_size": 100,
            "valid_splits": 1,
            "max_train_size": None,
            "gap": 20,
            "cross_validation": "GroupedTimeSeriesSplit",
        },
        "selection": {
            "proportion": 0,
            "criteria": "sharpe",
        },
        "train_targets": ["target_nomi_v4_20"],
        "train_endera": "0620",
        "validate_targets": [
            "target_nomi_v4_20",
        ],
        "validate_enderas": [
            "0620",
            "0720",
            "0820",
            "0920",
            "1020",
        ],
        "train_resample_freq": 1,
        "validate_resample_freq": 1,
        "output_folder": "numerai_models_v4_thesis",
        "model_no_start": 2000,
        "no_models_per_config": 10,
        "feature_sets": "v4-all",
    },
}


## Data Folders
MODEL_FOLDER = "numerai_models_v4_thesis"
PERFORMANCES_FOLDER = "numerai_models_performances_v4_thesis"
if not os.path.exists(f"{MODEL_FOLDER}/"):
    os.mkdir(f"{MODEL_FOLDER}/")
if not os.path.exists(f"{PERFORMANCES_FOLDER}/"):
    os.mkdir(f"{PERFORMANCES_FOLDER}/")

numerai_files = {
    "dataset": "data/v4_all_int8.parquet",
    "feature_corr": "data/v4_feature_corr.parquet",
    "feature_metadata": "data/v4_features.json",
}


def run_numerai_classic(optimisation_args):

    ### Logging
    log_file_path = f'{optimisation_args["ml_method"]["method"]}_{optimisation_args["model_params"]["model_no_start"]}.log'
    import logging

    logger = logging.getLogger("Numerai")
    logger.setLevel(level=logging.INFO)
    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename=log_file_path)
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(level=logging.INFO)
    logger.addHandler(fileHandler)

    #### Save Search Space
    search_space_file_path = f'search_space_{optimisation_args["ml_method"]["method"]}_{optimisation_args["model_params"]["model_no_start"]}.json'
    if not os.path.exists(search_space_file_path):
        with open(search_space_file_path, "w") as f:
            json.dump(optimisation_args, f)

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

    ### Get Model Performances in validation period
    feature_corr = pd.read_parquet(numerai_files["feature_corr"])
    with open(numerai_files["feature_metadata"], "r") as f:
        feature_metadata = json.load(f)
    features_optimizer = feature_metadata["feature_sets"]["fncv3_features"]

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
                feature_corr,
                features_optimizer,
                PERFORMANCES_FOLDER,
                data_file=numerai_files["dataset"],
                data_version="v4-all",
                target_col=["target_nomi_v4_20"],
            )


## Which Optimisation to run
BENCHMARKS = [
    optuna_mlp_args_v2,
]

for optimisation_args in BENCHMARKS:
    run_numerai_classic(optimisation_args)
