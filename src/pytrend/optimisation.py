#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Optimising hyper-parameters for ML models with Optuna
#


import pandas as pd
import numpy as np
import joblib, json, os, gc

import optuna
from optuna.samplers import RandomSampler, TPESampler


from .util import GroupedTimeSeriesSplit, strategy_metrics
from .benchmark import benchmark_pipeline, save_best_model, load_best_model
from .numerai import load_numerai_data, score_numerai


import logging

logger = logging.getLogger("Numerai")

### Create Hyper-parameter space for optuna
### Extract parameter space that needs to be optimised from the config dictionary
def create_optuna_space(config_dictionary, trial):
    space = dict()
    for step in ["feature_eng", "ml_method"]:
        for k, v in config_dictionary[step]["parameters"].items():
            if isinstance(v, list):
                space[k] = getattr(trial, f"suggest_{v[0]}")(name=k, **v[1])
            else:
                space[k] = v
    return space


### Create Parameter Sets from optuna trial instances
def create_parameters_sets(
    args,
    config_dictionary,
    seed=0,
):

    ### Feature Engineering
    feature_eng_parameters = {}
    for k, v in config_dictionary["feature_eng"]["parameters"].items():
        feature_eng_parameters[k] = args.get(k, v)

    ### ML Methods
    tabular_hyper = {
        "seed": seed,
    }

    for k, v in config_dictionary["ml_method"]["parameters"].items():
        tabular_hyper[k] = args.get(k, v)

    ### Additional Hyper-parameters to be passed to training loop, NOT used now
    additional_hyper = dict()

    return feature_eng_parameters, tabular_hyper, additional_hyper


# Create Objective function using optuna for Numerai Classic and Numerai Signals Tournament


def create_optuna_numerai_objective(
    config_dictionary, numerai_files, seed=0, debug=False
):
    def objective(trial):

        with open(numerai_files["feature_metadata"], "r") as f:
            feature_metadata = json.load(f)
        if config_dictionary["model_params"]["feature_sets"] == "v4":
            features_optimizer = feature_metadata["feature_sets"]["fncv3_features"]
            bad_features = [
                "feature_palpebral_univalve_pennoncel",
                "feature_unsustaining_chewier_adnoun",
                "feature_brainish_nonabsorbent_assurance",
                "feature_coastal_edible_whang",
                "feature_disprovable_topmost_burrower",
                "feature_trisomic_hagiographic_fragrance",
                "feature_queenliest_childing_ritual",
                "feature_censorial_leachier_rickshaw",
                "feature_daylong_ecumenic_lucina",
                "feature_steric_coxcombic_relinquishment",
            ]
            features_optimizer = list(set(features_optimizer) - set(bad_features))
        else:
            features_optimizer = list()

        features, targets, groups, weights = load_numerai_data(
            numerai_files["dataset"],
            feature_metadata=numerai_files["feature_metadata"],
            resample=0,
            resample_freq=config_dictionary["model_params"]["train_resample_freq"],
            target_col=config_dictionary["model_params"]["train_targets"],
            data_version=config_dictionary["model_params"]["feature_sets"],
            startera=config_dictionary["model_params"]["train_startera"],
            endera=config_dictionary["model_params"]["train_endera"],
        )

        param = create_optuna_space(config_dictionary, trial)

        logger.info(param)

        (
            feature_eng_parameters,
            tabular_hyper,
            additional_hyper,
        ) = create_parameters_sets(
            param,
            config_dictionary,
            seed=seed,
        )

        model_performance, trained_models, data, parameters = benchmark_pipeline(
            features,
            targets,
            weights,
            groups,
            feature_eng=config_dictionary["feature_eng"]["method"],
            feature_eng_parameters=feature_eng_parameters,
            tabular_model=config_dictionary["ml_method"]["method"],
            tabular_hyper=tabular_hyper,
            model_params=config_dictionary["model_params"]["train"],
            additional_hyper=additional_hyper,
            debug=debug,
        )
        ## Get Predictions for each of the walk forward model
        ## Score on Validation data
        predictions = list()
        for model_name in list(data.keys()):
            predictions.append(data[model_name]["prediction"])
        train_prediction_df = pd.DataFrame(pd.concat(predictions, axis=0).mean(axis=1))
        train_prediction_df.columns = ["prediction"]
        train_prediction_df["target"] = targets.reindex(train_prediction_df.index)
        train_prediction_df["era"] = groups.reindex(train_prediction_df.index)
        train_prediction_df, correlations_by_era = score_numerai(
            train_prediction_df,
            features,
            riskiest_features=features_optimizer,
            proportion=float(
                config_dictionary["model_params"]["selection"]["proportion"]
            ),
            era_col="era",
            target_col_name="target",
        )
        performances = strategy_metrics(correlations_by_era["neutralised_correlation"])
        metric = performances[
            config_dictionary["model_params"]["selection"]["criteria"]
        ]
        logger.info(f"Out of Sample Metric {metric}")
        return metric

    return objective


def optuna_search(
    config_dictionary,
    numerai_files,
    n_trials=10,
    timeout=10000,
    seed=0,
    debug=False,
):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    numerai_objective = create_optuna_numerai_objective(
        config_dictionary, numerai_files, seed=seed, debug=debug
    )
    study = optuna.create_study(
        direction="maximize",
    )
    study.optimize(
        numerai_objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True
    )

    return study.best_trial.params, study.best_trial.value


def train_best_model_optuna(
    target_col_name,
    end_era,
    best_parameters,
    config_dictionary,
    numerai_files,
    seed=0,
    debug=False,
):

    resample_seed = int(
        seed % config_dictionary["model_params"]["validate_resample_freq"]
    )
    features, targets, groups, weights = load_numerai_data(
        numerai_files["dataset"],
        feature_metadata=numerai_files["feature_metadata"],
        resample=resample_seed,
        resample_freq=config_dictionary["model_params"]["validate_resample_freq"],
        target_col=[target_col_name],
        data_version=config_dictionary["model_params"]["feature_sets"],
        startera=config_dictionary["model_params"]["train_startera"],
        endera=end_era,
    )

    output_folder = config_dictionary["model_params"]["output_folder"]

    if not os.path.exists(f"{output_folder}/"):
        os.mkdir(f"{output_folder}/")

    feature_eng_parameters, tabular_hyper, additional_hyper = create_parameters_sets(
        best_parameters,
        config_dictionary,
        seed=seed,
    )

    model_performance, trained_models, data, parameters = benchmark_pipeline(
        features,
        targets,
        weights,
        groups,
        feature_eng=config_dictionary["feature_eng"]["method"],
        feature_eng_parameters=feature_eng_parameters,
        tabular_model=config_dictionary["ml_method"]["method"],
        tabular_hyper=tabular_hyper,
        model_params=config_dictionary["model_params"]["validate"],
        additional_hyper=additional_hyper,
        debug=debug,
    )

    ## Save each model
    for model_name in list(trained_models.keys()):
        ## Save Parameters and Feature Transformer
        output_parameters_path = f"{output_folder}/{model_name}_{seed}.parameters"
        output_parameters = dict()
        output_parameters["parameters"] = parameters[model_name]
        output_parameters["transformer"] = trained_models[model_name]["transformer"]
        joblib.dump(output_parameters, output_parameters_path)

        ## Save Model
        output_model_path = f"{output_folder}/{model_name}_{seed}.model"
        save_best_model(
            trained_models[model_name]["model"],
            parameters[model_name]["model"]["tabular_model"],
            output_model_path,
        )

    return None


def numerai_optimisation_pipeline_optuna(
    config_dictionary,
    numerai_files,
    run_optimisation=True,
    optimised_parameters_path="numerai_best_parameters.json",
    grid_search_seed=0,
    n_trials=40,
    timeout=2000,
    debug=False,
):

    ## Search for optimal hyper-parameters
    if run_optimisation:
        best_parameters, best_value = optuna_search(
            config_dictionary,
            numerai_files,
            seed=grid_search_seed,
            n_trials=n_trials,
            timeout=timeout,
            debug=debug,
        )
        with open(optimised_parameters_path, "w") as f:
            best_parameters["Optuna_Best_Value"] = best_value
            json.dump(best_parameters, f)
    else:
        with open(optimised_parameters_path, "r") as f:
            best_parameters = json.load(f)
            logger.info(f"Using Best parameters {best_parameters}")

    START_SEED = config_dictionary["model_params"]["model_no_start"]
    NO_MODELS_PER_CONFIG = config_dictionary["model_params"]["no_models_per_config"]

    if config_dictionary["model_params"]["mix_cv"]:
        for target_col_name in config_dictionary["model_params"]["validate_targets"]:
            for end_era in config_dictionary["model_params"]["validate_enderas"]:
                for seed in range(START_SEED, START_SEED + NO_MODELS_PER_CONFIG):
                    ## Check if Model already exists
                    output_folder = config_dictionary["model_params"]["output_folder"]
                    tabular_model = config_dictionary["ml_method"]["method"]
                    feature_eng = config_dictionary["feature_eng"]["method"]
                    model_name = "{}_{}_{}".format(tabular_model, feature_eng, 1)
                    output_model_path = f"{output_folder}/{model_name}_{seed}.model"
                    if not os.path.exists(output_model_path):
                        train_best_model_optuna(
                            target_col_name,
                            end_era,
                            best_parameters,
                            config_dictionary,
                            numerai_files,
                            seed=seed,
                            debug=debug,
                        )

                START_SEED = START_SEED + NO_MODELS_PER_CONFIG
    else:
        for end_era in config_dictionary["model_params"]["validate_enderas"]:
            for target_col_name in config_dictionary["model_params"][
                "validate_targets"
            ]:
                for seed in range(START_SEED, START_SEED + NO_MODELS_PER_CONFIG):
                    ## Check if Model already exists
                    output_folder = config_dictionary["model_params"]["output_folder"]
                    tabular_model = config_dictionary["ml_method"]["method"]
                    feature_eng = config_dictionary["feature_eng"]["method"]
                    model_name = "{}_{}_{}".format(tabular_model, feature_eng, 1)
                    output_model_path = f"{output_folder}/{model_name}_{seed}.model"
                    if not os.path.exists(output_model_path):
                        train_best_model_optuna(
                            target_col_name,
                            end_era,
                            best_parameters,
                            config_dictionary,
                            numerai_files,
                            seed=seed,
                            debug=debug,
                        )

                START_SEED = START_SEED + NO_MODELS_PER_CONFIG
