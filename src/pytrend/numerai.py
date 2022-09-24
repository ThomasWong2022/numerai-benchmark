#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of tools for data processing for Numerai and other temporal tabular data competitions
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
import scipy

import joblib, datetime, json

import xgboost

from .benchmark import load_best_model, save_best_model
from .util import align_features_target, RollingTSTransformer, GroupedTimeSeriesSplit


## Shifting Numerai Era
def convert_era(era, gap=6):
    new_era_int = int(era) + gap
    new_era = str(new_era_int)
    while len(new_era) < 4:
        new_era = "0" + new_era
    return new_era


### Map Numerai Era to actual datetime
def create_era_index(
    df,
    baseline=datetime.datetime(year=2003, month=1, day=3),
    replace_col=False,
    era_col="era",
):
    delete_col = False
    if not era_col in df.columns:
        df[era_col] = df.index
        delete_col = True
    if df[era_col].dtypes == "<M8[ns]":
        return df
    mapped_era = [
        baseline + datetime.timedelta(days=7 * (x - 1)) for x in df[era_col].apply(int)
    ]
    if not replace_col:
        df.index = mapped_era
    else:
        df[era_col] = mapped_era
    if delete_col:
        df.drop(era_col, axis=1, inplace=True)
    return df


def load_numerai_data(
    filename,
    feature_metadata="data/v4_features.json",
    resample=0,
    resample_freq=1,
    target_col=["target"],
    era_col="era",
    data_version="v4",
    feature_selection="v4",
    startera="0001",
    endera="9999",
):
    ## Read Train Data
    df_raw = pd.read_parquet(filename)

    ## Work Around for live with era with unknown era
    df_raw[era_col] = df_raw[era_col].replace("X", "9999")

    ## Downsample Data
    df = df_raw[(df_raw[era_col] >= startera) & (df_raw[era_col] <= endera)].iloc[
        resample::resample_freq
    ]

    del df_raw

    ## Features Sets
    if data_version == "v4":
        feature_col = [col for col in df.columns if col.startswith("feature_")]

        if feature_selection == "v3":
            with open(feature_metadata, "r") as f:
                feature_metadata = json.load(f)
            feature_col = feature_metadata["feature_sets"]["v3_equivalent_features"]

        if feature_selection == "v2":
            with open(feature_metadata, "r") as f:
                feature_metadata = json.load(f)
            feature_col = feature_metadata["feature_sets"]["v2_equivalent_features"]

    if data_version == "v3":
        feature_col = [col for col in df.columns if col.startswith("feature_")]

    ## Features and Targets are DataFrame
    features = df[feature_col].fillna(2) - 2
    targets = df[target_col].fillna(0.5) - 0.5
    ## Group column has to be pd.Series for time-series cross validation
    groups = df[era_col]
    ## weights column has to be pd.Series for time-series cross validation
    df["weights"] = 1
    weights = df["weights"]
    return features, targets, groups, weights


"""
Generate Predictions for Numerai 

trained_model: model object which has method .predict to generate predictions
parameters: dictionary which contains parameters of the trained_model
modelname: str Name of Model
start_iteration: for tree-based methods, skip the first N trees in model when generating predictions 
startera: first era to get predictions 
endera: last era to get predictions 


Output: prediction_df: pd.DataFrame with columns era, prediction, model_name, target_col 

"""


def predict_numerai(
    features_raw,
    targets,
    groups,
    trained_model,
    parameters,
    modelname="sample",
    gbm_start_iteration=None,  ## Backward Comptability
    era_col="era",
    debug=False,
):
    ## Score on Dataset

    selected_cols = parameters["parameters"]["model"]["feature_columns"]
    target_col = parameters["parameters"]["model"]["target_columns"]

    ## Transform Features
    if parameters["transformer"] is not None:
        transformer = parameters["transformer"]
        features = transformer.transform(features_raw[selected_cols], is_train=False)
    else:
        features = features_raw[selected_cols]

    ## Run Predictions
    ## For tree-based models can run some of the trees only
    if parameters["parameters"]["model"]["tabular_model"] in [
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
        "lightgbm-rf",
    ]:
        ## Backward Compatability
        if "additional" in parameters["parameters"] and gbm_start_iteration is None:
            gbm_start_iteration = parameters["parameters"]["additional"].get(
                "gbm_start_iteration", 0
            )
        valid_iteration = min(gbm_start_iteration, int(trained_model.num_trees() // 2))
        predictions_raw = trained_model.predict(
            features, start_iteration=valid_iteration
        )
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "xgboost-dart",
        "xgboost-gbtree",
    ]:
        ## Backward Compatability
        if "additional" in parameters["parameters"] and gbm_start_iteration is None:
            gbm_start_iteration = parameters["parameters"]["additional"].get(
                "gbm_start_iteration", 0
            )
        end_iteration = trained_model.get_booster().best_iteration
        valid_iteration = min(gbm_start_iteration, int(end_iteration // 2))
        predictions_raw = trained_model.predict(
            features,
            iteration_range=(valid_iteration, end_iteration),
        )
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "pytorch-tabular-tabtransformer",
        "pytorch-tabular-node",
        "pytorch-tabular-categoryembedding",
    ]:
        predictions_raw = trained_model.predict(features)[
            f"{target_col[0]}_prediction"
        ].values
    elif parameters["parameters"]["model"]["tabular_model"] == "tabnet":
        predictions_raw = trained_model.predict(features.values)
    else:
        predictions_raw = trained_model.predict(features)

    ## Process Predictions
    predictions = pd.DataFrame(
        predictions_raw,
        columns=target_col,
        index=targets.index,
    )
    predictions[era_col] = groups
    ## Rank Predictions within each era
    normalised_predictions = list()
    for i, df in predictions.groupby(era_col):
        per_era = df[target_col].rank(pct=True, axis=0)
        normalised_predictions.append(per_era)
    processed_predictions = pd.concat(normalised_predictions, axis=0)
    predictions["prediction"] = processed_predictions[target_col].mean(axis=1)
    prediction_df = pd.concat([predictions[[era_col, "prediction"]], targets], axis=1)
    prediction_df["model_name"] = modelname
    return prediction_df


###


def predict_numerai_dynamic(
    features_raw,
    targets,
    groups,
    trained_model,
    parameters,
    correlation_matrix,  ## dyanmic feature augmentation
    modelname="sample",
    gbm_start_iteration=None,  ## Backward Comptability
    era_col="era",
    debug=False,
    cutoff=420,  ## dyanmic feature augmentation
    gap=6,  ## dyanmic feature augmentation
    lookback=52,  ## dyanmic feature augmentation
):

    ## Generate Feature Momentum Leaderboard
    factor_momentum = correlation_matrix.shift(gap).fillna(0).rolling(lookback).mean()
    factor_volatility = correlation_matrix.shift(gap).fillna(0).rolling(lookback).std()

    fm_max_index = factor_momentum.index.max()
    fm_min_index = factor_momentum.index.min()

    ##
    factor_momentum_eras = factor_momentum.unstack(level=0).reset_index()
    factor_momentum_eras.columns = ["feature_name", "era", "momentum"]
    factor_volatility_eras = factor_volatility.unstack(level=0).reset_index()
    factor_volatility_eras.columns = ["feature_name", "era", "volatility"]

    ## Get index by era
    prediction_dynamic = list()

    for i, df in pd.DataFrame(groups).groupby(era_col):

        features_era = features_raw.loc[df.index]
        targets_era = targets.loc[df.index]
        groups_era = groups.loc[df.index]

        if debug:
            print(i, df.shape)

        if (i <= fm_max_index) & (i >= fm_min_index):

            factor_momentum_era = factor_momentum_eras[factor_momentum_eras["era"] == i]
            factor_volatility_era = factor_volatility_eras[
                factor_volatility_eras["era"] == i
            ]

            ## Momentum Winners
            momentum_winners = factor_momentum_era.sort_values("momentum").tail(cutoff)[
                "feature_name"
            ]
            features_era_augmented = features_era.copy()
            features_era_augmented[momentum_winners] = np.nan
            prediction_df_era = predict_numerai(
                features_era_augmented,
                targets_era,
                groups_era,
                trained_model,
                parameters,
                modelname=f"{modelname}-momentum_winners",
                gbm_start_iteration=gbm_start_iteration,  ## Backward Comptability
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era)
            ## Momentum Losers
            momentum_losers = factor_momentum_era.sort_values("momentum").head(cutoff)[
                "feature_name"
            ]
            features_era_augmented = features_era.copy()
            features_era_augmented[momentum_losers] = np.nan
            prediction_df_era = predict_numerai(
                features_era_augmented,
                targets_era,
                groups_era,
                trained_model,
                parameters,
                modelname=f"{modelname}-momentum_losers",
                gbm_start_iteration=gbm_start_iteration,  ## Backward Comptability
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era)
            ## High Volatility
            high_vol = factor_volatility_era.sort_values("volatility").tail(cutoff)[
                "feature_name"
            ]
            features_era_augmented = features_era.copy()
            features_era_augmented[high_vol] = np.nan
            prediction_df_era = predict_numerai(
                features_era_augmented,
                targets_era,
                groups_era,
                trained_model,
                parameters,
                modelname=f"{modelname}-high_vol",
                gbm_start_iteration=gbm_start_iteration,  ## Backward Comptability
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era)
            ## Low Volatility
            low_vol = factor_volatility_era.sort_values("volatility").head(cutoff)[
                "feature_name"
            ]
            features_era_augmented = features_era.copy()
            features_era_augmented[low_vol] = np.nan
            prediction_df_era = predict_numerai(
                features_era_augmented,
                targets_era,
                groups_era,
                trained_model,
                parameters,
                modelname=f"{modelname}-low_vol",
                gbm_start_iteration=gbm_start_iteration,  ## Backward Comptability
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era)

            prediction_df_era = predict_numerai(
                features_era,
                targets_era,
                groups_era,
                trained_model,
                parameters,
                modelname=f"{modelname}-baseline",
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era)

    return pd.concat(prediction_dynamic, axis=0)


"""
prediction_df: pd.DataFrame with columns era, prediction, model_name, target_col  and index id 
features: pd.DataFrame with columns feature_xxx and index id 
riskiest_fatures: list of str 


Output 
prediction_df: pd.DataFrame with columns era, model_name, prediction, neutralised_prediction, target_col, index id 
correlations_by_era: pd.DataFrame with columns correlation, normalised_correlation, neutralised_correlation, index era 

"""


def score_numerai(
    prediction_df,
    features,
    riskiest_features,
    proportion=0,
    modelname="sample",
    target_col_name="target",
    prediction_col="prediction",
    era_col="era",
    debug=False,
):
    ## Find Correlation by era
    correlations_by_era = list()
    for i, df in prediction_df.groupby(era_col):
        output = dict()
        output[era_col] = i
        ## Correlation (Corr)
        output["correlation"] = np.corrcoef(
            df[target_col_name], df[prediction_col].rank(pct=True)
        )[0, 1]
        ## Normalise prediction
        temp = (scipy.stats.rankdata(df[prediction_col], method="ordinal") - 0.5) / len(
            df[prediction_col]
        )
        df["normalised_prediction"] = scipy.stats.norm.ppf(temp)
        #  output["normalised_correlation"] = np.corrcoef(df[target_col_name], df["normalised_prediction"])[0, 1]
        ## Neutralised targets (FNC)
        if proportion > 0 and len(riskiest_features) > 0:
            exposures = features.loc[df.index, riskiest_features]
            df["neutralised_prediction"] = df[
                "normalised_prediction"
            ] - proportion * exposures.dot(
                np.linalg.pinv(exposures).dot(df["normalised_prediction"])
            )
            df["neutralised_prediction"] = (
                df["neutralised_prediction"] / df["neutralised_prediction"].std()
            )
            output["neutralised_correlation"] = np.corrcoef(
                df[target_col_name], df["neutralised_prediction"].rank(pct=True)
            )[0, 1]
            prediction_df.loc[df.index, "neutralised_prediction"] = df[
                "neutralised_prediction"
            ].rank(pct=True)
        else:
            output["neutralised_correlation"] = output["correlation"]
            prediction_df.loc[df.index, "neutralised_prediction"] = df[
                prediction_col
            ].rank(pct=True)
        correlations_by_era.append(output)
        if debug:
            print(output)
    correlations_by_era = pd.DataFrame.from_records(correlations_by_era)
    prediction_df["model_name"] = modelname
    correlations_by_era["model_name"] = modelname
    return prediction_df, correlations_by_era


### Perform Dynamic Feature Neutralisation given predicted values from ML models and score the neutralised predictions
def dynamic_feature_neutralisation(
    prediction_df,
    features_raw,
    correlation_matrix,
    features_optimizer=None,
    modelname="sample",
    gbm_start_iteration=None,
    era_col="era",
    cutoff=420,
    gap=6,
    lookback=52,
    proportion=1,
    debug=False,
):

    if features_optimizer is None:
        features_optimizer = features_raw.columns[:cutoff]

    ## Generate Feature Momentum Leaderboard
    factor_momentum = correlation_matrix.shift(gap).fillna(0).rolling(lookback).mean()
    factor_volatility = correlation_matrix.shift(gap).fillna(0).rolling(lookback).std()

    fm_max_index = factor_momentum.index.max()
    fm_min_index = factor_momentum.index.min()

    ##
    factor_momentum_eras = factor_momentum.unstack(level=0).reset_index()
    factor_momentum_eras.columns = ["feature_name", "era", "momentum"]
    factor_volatility_eras = factor_volatility.unstack(level=0).reset_index()
    factor_volatility_eras.columns = ["feature_name", "era", "volatility"]

    ## Get index by era
    prediction_dynamic = list()
    correlation_dynamic = list()
    for i, df in prediction_df.groupby(era_col):
        if debug:
            print(i, df.shape)
        if (i <= fm_max_index) & (i >= fm_min_index):
            prediction_df_era = prediction_df.loc[df.index]
            features_raw_era = features_raw.loc[df.index]
            factor_momentum_era = factor_momentum_eras[factor_momentum_eras["era"] == i]
            factor_volatility_era = factor_volatility_eras[
                factor_volatility_eras["era"] == i
            ]

            ## Baseline
            prediction_df_era_new, correlations_by_era = score_numerai(
                prediction_df_era,
                features_raw_era,
                None,
                proportion=0,
                modelname=f"{modelname}-baseline",
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era_new.copy())
            correlation_dynamic.append(correlations_by_era)

            ## Optimizer
            prediction_df_era_new, correlations_by_era = score_numerai(
                prediction_df_era,
                features_raw_era,
                features_optimizer,
                proportion=proportion,
                modelname=f"{modelname}-optimizer",
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era_new.copy())
            correlation_dynamic.append(correlations_by_era)

            DFN_params = [
                (
                    "momentum",
                    420,
                    "tail",
                    "momentum_winners",
                ),
                (
                    "momentum",
                    420,
                    "head",
                    "momentum_losers",
                ),
                (
                    "volatility",
                    420,
                    "tail",
                    "high_vol",
                ),
                (
                    "volatility",
                    420,
                    "head",
                    "low_vol",
                ),
                (
                    "momentum",
                    105,
                    "tail",
                    "momentum_winners_small",
                ),
                (
                    "momentum",
                    105,
                    "head",
                    "momentum_losers_small",
                ),
                (
                    "volatility",
                    105,
                    "tail",
                    "high_vol_small",
                ),
                (
                    "volatility",
                    105,
                    "head",
                    "low_vol_small",
                ),
            ]

            for DFN_param in DFN_params:

                if DFN_param[0] == "momentum":
                    if DFN_param[2] == "tail":
                        selected_features = factor_momentum_era.sort_values(
                            "momentum"
                        ).tail(DFN_param[1])["feature_name"]
                    else:
                        selected_features = factor_momentum_era.sort_values(
                            "momentum"
                        ).head(DFN_param[1])["feature_name"]
                elif DFN_param[0] == "volatility":
                    if DFN_param[2] == "tail":
                        selected_features = factor_volatility_era.sort_values(
                            "volatility"
                        ).tail(DFN_param[1])["feature_name"]
                    else:
                        selected_features = factor_volatility_era.sort_values(
                            "volatility"
                        ).head(DFN_param[1])["feature_name"]

                prediction_df_era_new, correlations_by_era = score_numerai(
                    prediction_df_era,
                    features_raw_era,
                    selected_features,
                    proportion=proportion,
                    modelname=f"{modelname}-{DFN_param[3]}",
                    era_col=era_col,
                    debug=debug,
                )
                prediction_dynamic.append(prediction_df_era_new.copy())
                correlation_dynamic.append(correlations_by_era)

    return pd.concat(prediction_dynamic, axis=0), pd.concat(correlation_dynamic, axis=0)


"""
Factor Timing

rawdata: pd.DataFrame: Numerai dataset with columns containing the 1149 features and 20 targets, index id 



"""


def numerai_feature_correlation_matrix(
    rawdata, feature_col=None, target_col_name=None, era_col="era"
):

    if not target_col_name:
        target_col_name = "target"
    if not feature_col:
        feature_col = [x for x in rawdata.columns if x.startswith("feature_")]

    output = dict()
    for i, df in rawdata.groupby(era_col):
        corr_dict = dict()
        for feature in feature_col:
            corr_dict[feature] = np.corrcoef(df[feature], df[target_col_name])[0, 1]
        output[i] = corr_dict

    return pd.DataFrame.from_records(output).transpose()[feature_col]


##### Scoring Numerai Models
def score_numerai_multiple(
    Numerai_Model_Names,
    correlation_matrix=None,
    filename="data/v4_all_int8.parquet",
    data_version="v4",
    feature_selection="v4",
    startera="0001",
    endera="9999",
    riskiest_features=None,
    proportion=0,
    gbm_start_iteration=None,  ## Set to None to use the lgb_start_iteration from Numerai Hyper-parameter Search
    prediction_mode="default",
    score_mode="default",
    debug=False,
    era_col="era",
    target_col=["target"],
):

    features, targets, groups, weights = load_numerai_data(
        filename=filename,
        target_col=target_col,
        era_col=era_col,
        data_version=data_version,
        feature_selection=feature_selection,
        startera=startera,
        endera=endera,
    )

    prediction_df_list = list()
    score_df_list = list()

    for Numerai_Model_Name in Numerai_Model_Names:
        modelname = Numerai_Model_Name.replace(".parameters", ".model")
        parameters = joblib.load(Numerai_Model_Name)
        most_recent_model = load_best_model(
            parameters["parameters"]["model"]["tabular_model"], modelname
        )

        if debug:
            print(modelname)

        if prediction_mode == "default":
            prediction_df = predict_numerai(
                features,
                targets,
                groups,
                most_recent_model,
                parameters,
                modelname=modelname,
                gbm_start_iteration=gbm_start_iteration,
                era_col=era_col,
                debug=debug,
            )
            prediction_df_list.append(prediction_df)
        else:
            prediction_df = predict_numerai_dynamic(
                features,
                targets,
                groups,
                most_recent_model,
                parameters,
                correlation_matrix,
                modelname=modelname,
                gbm_start_iteration=gbm_start_iteration,
                era_col=era_col,
                debug=debug,
            )
            prediction_df_list.append(prediction_df)

        if riskiest_features is None:
            riskiest_features = parameters["parameters"]["model"]["feature_columns"]

        if debug:
            print(prediction_df.columns, prediction_df.shape)

        if score_mode == "default":
            if prediction_mode == "default":
                prediction_df, correlations_by_era = score_numerai(
                    prediction_df,
                    features,
                    riskiest_features,
                    0,
                    modelname,
                    target_col_name=target_col[0],
                    era_col=era_col,
                    debug=debug,
                )
                score_df_list.append(correlations_by_era)

            else:
                modelnames = prediction_df["model_name"].unique()
                for modelname in modelnames:
                    prediction_df_one = prediction_df[
                        prediction_df["model_name"] == modelname
                    ]
                    prediction_df_one, correlations_by_era = score_numerai(
                        prediction_df_one,
                        features,
                        riskiest_features,
                        0,
                        modelname,
                        target_col_name=target_col[0],
                        era_col=era_col,
                        debug=debug,
                    )
                    score_df_list.append(correlations_by_era)

    if len(prediction_df_list) > 0:
        if prediction_mode == "default":

            average_prediction_df = (
                pd.concat(prediction_df_list, axis=0)
                .groupby("id")[
                    [
                        era_col,
                        "prediction",
                    ]
                    + target_col
                ]
                .mean()
                .reindex(prediction_df_list[0].index)
            )
            average_prediction_df[era_col] = prediction_df_list[0][era_col]

            if debug:
                print(average_prediction_df.columns, average_prediction_df.shape)

            if score_mode == "default":
                average_prediction_df, correlations_by_era = score_numerai(
                    average_prediction_df,
                    features,
                    riskiest_features,
                    proportion,
                    "Mean",
                    target_col_name=target_col[0],
                    era_col=era_col,
                    debug=debug,
                )
                return (
                    average_prediction_df,
                    correlations_by_era,
                    prediction_df_list,
                    score_df_list,
                )
            else:
                return (
                    average_prediction_df,
                    None,
                    prediction_df_list,
                    None,
                )

        else:
            average_prediction_df_list = list()
            correlations_by_era_list = list()
            modelnames = prediction_df["model_name"].unique()
            for modelname in modelnames:
                temp = pd.concat(prediction_df_list, axis=0)
                prediction_df_one = temp[temp["model_name"] == modelname]
                average_prediction_df = (
                    prediction_df_one.groupby("id")[
                        [
                            era_col,
                            "prediction",
                        ]
                        + target_col
                    ]
                    .mean()
                    .reindex(prediction_df_list[0].index)
                )
                average_prediction_df[era_col] = prediction_df_list[0][era_col]
                average_prediction_df, correlations_by_era = score_numerai(
                    average_prediction_df,
                    features,
                    riskiest_features,
                    proportion,
                    "Mean",
                    target_col_name=target_col[0],
                    era_col=era_col,
                    debug=debug,
                )
                average_prediction_df_list.append(average_prediction_df)
                correlations_by_era_list.append(correlations_by_era)

            return (
                average_prediction_df_list,
                correlations_by_era_list,
                prediction_df_list,
                score_df_list,
            )

    else:
        return pd.DataFrame(), pd.DataFrame(), prediction_df_list, score_df_list


"""
Linear Factor Model

"""


def numerai_factor_portfolio(
    rawdata,
    correlation_matrix=None,
    feature_col=None,
    target_col_name=None,
    era_col="era",
    cutoff=420,
    gap=6,
    lookback=52,
):

    if not feature_col:
        feature_col = [x for x in rawdata.columns if x.startswith("feature_")]
    if not target_col_name:
        target_col_name = "target"

    if correlation_matrix is None:
        correlation_matrix = numerai_feature_correlation_matrix(
            rawdata, feature_col, target_col_name
        )

    ## Factor Momentum Portfolio
    factor_momentum = correlation_matrix.shift(gap).fillna(0).rolling(lookback).mean()
    factor_volatility = correlation_matrix.shift(gap).fillna(0).rolling(lookback).std()
    fm_max_index = factor_momentum.index.max()
    fm_min_index = factor_momentum.index.min()

    factor_momentum_eras = factor_momentum.unstack(level=0).reset_index()
    factor_momentum_eras.columns = ["feature_name", "era", "momentum"]
    factor_volatility_eras = factor_volatility.unstack(level=0).reset_index()
    factor_volatility_eras.columns = ["feature_name", "era", "volatility"]

    prediction_dynamic = list()
    correlation_dynamic = list()

    for i, df in rawdata.groupby(era_col):

        if (i <= fm_max_index) & (i >= fm_min_index):

            ## Equal Weighted
            portfolio_predictions = df[[era_col, target_col_name]]
            portfolio_predictions["prediction"] = df[feature_col].mean(axis=1)

            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="equal_weighted",
                target_col_name=target_col_name,
            )
            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)

            ## Factor Momentum
            portfolio_predictions = df[[era_col, target_col_name]]
            per_era = df[feature_col] * np.sign(factor_momentum.loc[i, feature_col])
            portfolio_predictions["prediction"] = per_era.mean(axis=1)

            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="factor_momentum",
                target_col_name=target_col_name,
            )

            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)

            factor_momentum_era = factor_momentum_eras[factor_momentum_eras["era"] == i]
            factor_volatility_era = factor_volatility_eras[
                factor_volatility_eras["era"] == i
            ]

            momentum_winners = factor_momentum_era.sort_values("momentum").tail(cutoff)[
                "feature_name"
            ]
            momentum_losers = factor_momentum_era.sort_values("momentum").head(cutoff)[
                "feature_name"
            ]
            high_vol = factor_volatility_era.sort_values("volatility").tail(cutoff)[
                "feature_name"
            ]
            low_vol = factor_volatility_era.sort_values("volatility").head(cutoff)[
                "feature_name"
            ]

            ## Recent Winners
            portfolio_predictions = df[[era_col, target_col_name]]
            per_era = df[momentum_winners] * np.sign(
                factor_momentum.loc[i, momentum_winners]
            )
            portfolio_predictions["prediction"] = per_era.mean(axis=1)
            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="recent_winners",
                target_col_name=target_col_name,
            )

            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)

            ## Recent Losers
            portfolio_predictions = df[[era_col, target_col_name]]
            per_era = df[momentum_losers] * np.sign(
                factor_momentum.loc[i, momentum_losers]
            )
            portfolio_predictions["prediction"] = per_era.mean(axis=1)
            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="recent_losers",
                target_col_name=target_col_name,
            )
            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)

            ## High Vol
            portfolio_predictions = df[[era_col, target_col_name]]
            per_era = df[high_vol] * np.sign(factor_momentum.loc[i, high_vol])
            portfolio_predictions["prediction"] = per_era.mean(axis=1)
            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="high_vol",
                target_col_name=target_col_name,
            )
            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)
            ## Low Vol
            portfolio_predictions = df[[era_col, target_col_name]]
            per_era = df[low_vol] * np.sign(factor_momentum.loc[i, low_vol])
            portfolio_predictions["prediction"] = per_era.mean(axis=1)
            prediction_era, correlations_era = score_numerai(
                portfolio_predictions,
                df[feature_col],
                None,
                proportion=0,
                modelname="low_vol",
                target_col_name=target_col_name,
            )
            prediction_dynamic.append(prediction_era)
            correlation_dynamic.append(correlations_era)

    return pd.concat(prediction_dynamic, axis=0), pd.concat(correlation_dynamic, axis=0)


"""
Finetune Model 


"""


def predict_numerai_online(
    features_raw,
    targets,
    groups,
    weights,
    trained_model,
    parameters,
    model_params=None,
    riskiest_features=None,
    proportion=0,
    modelname="sample",
    era_col="era",
    target_col_name="target",
    debug=False,
):
    ## Score on Dataset

    selected_cols = parameters["parameters"]["model"]["feature_columns"]
    target_col = parameters["parameters"]["model"]["target_columns"]

    ## Start Cross Validation
    if model_params is None:
        model_params = {
            "valid_splits": 5,
            "test_size": 52 * 1,
            "max_train_size": None,
            "gap": 16,
        }

    tscv = GroupedTimeSeriesSplit(
        valid_splits=model_params["valid_splits"],
        test_size=model_params["test_size"],
        max_train_size=model_params["max_train_size"],
        gap=model_params["gap"],
    )

    prediction_df_list = list()
    correlation_df_list = list()

    for train_index, test_index in tscv.split(features_raw, groups=groups):

        ## Get Trained and Test Data
        X_train, X_test = (
            features_raw.loc[train_index, selected_cols],
            features_raw.loc[test_index, selected_cols],
        )
        y_train, y_test = targets.loc[train_index, :], targets.loc[test_index, :]
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

        ## Transform Features
        if parameters["transformer"] is not None:
            transformer = parameters["transformer"]
            features_train = transformer.transform(X_train, is_train=False)
        else:
            features_train = X_train

        tabular_model = parameters["parameters"]["model"]["tabular_model"]

        ## Refit Existing lightgbm models, update tree values by retain tree structures
        if tabular_model in [
            "lightgbm-gbdt",
            "lightgbm-dart",
            "lightgbm-goss",
            "lightgbm-rf",
        ]:

            trained_model.refit(features_train, y_train)

        ## Refit Existing lightgbm models, update tree values by retain tree structures
        ## The current API is not working
        if tabular_model in [
            "xgboost-dart",
            "xgboost-gbtree",
        ]:
            trained_model.set_params(
                process_type="update",
                updater="refresh",
                refresh_leaf=True,
            )

            trained_model.fit(features_train, y_train)

        ## Predict and Score on validation

        if not riskiest_features:
            riskiest_features = (X_test.columns,)

        prediction_df = predict_numerai(
            X_test,
            y_test,
            group_test,
            trained_model,
            parameters,
            modelname=modelname,
            era_col=era_col,
            debug=debug,
        )

        prediction_df, correlations_by_era = score_numerai(
            prediction_df,
            X_test,
            riskiest_features,
            proportion=proportion,
            modelname=modelname,
            target_col_name=target_col_name,
            prediction_col="prediction",
            era_col=era_col,
            debug=debug,
        )

        prediction_df_list.append(prediction_df)
        correlation_df_list.append(correlations_by_era)

    prediction_df_all = pd.concat(prediction_df_list, axis=0)
    correlation_df_all = pd.concat(correlation_df_list, axis=0)

    return trained_model, prediction_df_all, correlation_df_all
