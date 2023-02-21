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

import joblib, datetime, json, os
import pandas as pd
import numpy as np
import scipy
import torch
import xgboost

from .benchmark import load_best_model
from .util import strategy_metrics
from .feature import NumeraiTransformer, NumeraiTransformerV4


if torch.cuda.is_available():
    import cupy as cp
    import cudf
    from cuml.neighbors import KNeighborsRegressor
else:
    from sklearn.neighbors import KNeighborsRegressor


"""
Helper Functions to convert Numerai Era and Datetime 
"""

## Shifting Numerai Era
def shift_era(era, gap=6):
    new_era_int = int(era) + gap
    new_era = str(new_era_int)
    while len(new_era) < 4:
        new_era = "0" + new_era
    return new_era


## Convert datetime into Numerai eras
def convert_datetime_to_era(sample_date):
    baseline = datetime.datetime(year=2003, month=1, day=3)
    differences = datetime.datetime.strptime(sample_date, "%Y-%m-%d") - baseline
    new_era = str(differences.days // 7 + 1)
    while len(new_era) < 4:
        new_era = "0" + new_era
    return new_era


def convert_era_to_datetime(era):
    baseline = datetime.datetime(year=2003, month=1, day=3)
    new_datetime = baseline + datetime.timedelta(days=7 * (int(era) - 1))
    return new_datetime


### Map columns Numerai Era to datetime
def create_era_index(
    df,
    baseline=datetime.datetime(year=2003, month=1, day=3),
):
    mapped_era = [
        baseline + datetime.timedelta(days=7 * (int(x) - 1)) for x in df.index
    ]
    df.index = mapped_era
    return df


"""
Data Loader for Numerai Data 

"""


def load_numerai_data_era(
    filename,
    feature_metadata="v4_features.json",
    resample=0,
    resample_freq=1,
    target_col=["target"],
    era_col="era",
    data_version="v4",
    startera=None,
    endera=None,
):
    ## Read Train Data
    df_raw = pd.read_parquet(filename)
    ## Select Range
    if startera is not None and endera is not None:
        df_raw = df_raw[(df_raw[era_col] <= endera) & (df_raw[era_col] >= startera)]
    elif endera is not None:
        df_raw = df_raw[(df_raw[era_col] <= endera)]
    ## Downsample Eras
    if resample_freq > 1:
        downsampled_eras = df_raw[era_col].unique()[resample::resample_freq]
        df = df_raw[df_raw[era_col].isin(downsampled_eras)]
    else:
        df = df_raw.copy()

    del df_raw

    ## Features Sets
    feature_col = [col for col in df.columns if col.startswith("feature_")]

    if data_version in [
        "v4",
        "v4-all",
    ]:
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
        feature_col = list(set(feature_col) - set(bad_features))

    ## Features and Targets are DataFrame
    if data_version == "signals":
        features = df[feature_col].fillna(0)
    ## For Numerai Classic Tournament, v4 dataset
    else:
        features = df[feature_col].fillna(2) - 2
    
    target_median = df[target_col].median()
    targets = df[target_col].fillna(target_median) - target_median
    ## Group column has to be pd.Series for time-series cross validation
    groups = df[era_col]
    ## weights column has to be pd.Series for time-series cross validation
    df["weights"] = 1
    weights = df["weights"]
    return features.astype(np.int8), targets.astype(np.float32), groups, weights


def load_numerai_data(
    data_folder,
    feature_metadata="v4_features.json",
    resample=0,
    resample_freq=1,
    target_col=["target"],
    era_col="era",
    data_version="v4",
    startera=None,
    endera=None,
):

    if data_version in [
        "v4",
        "v4.1",
        "v5",
        "v6",
    ]:

        features_list = list()
        targets_list = list()
        groups_list = list()
        weights_list = list()

        if startera is None:
            startera = "0001"
        if endera is None:
            endera = "0001"

        for i in range(int(startera) + resample, int(endera) + 1, resample_freq):
            if i <= 9:
                test_start_str = "000" + str(i)
            elif i <= 99:
                test_start_str = "00" + str(i)
            elif i <= 999:
                test_start_str = "0" + str(i)
            else:
                test_start_str = str(i)

            data_file = f"{data_folder}/{data_version}_{test_start_str}_int8.parquet"

            features, targets, groups, weights = load_numerai_data_era(
                data_file,
                feature_metadata=feature_metadata,
                resample=0,
                resample_freq=1,
                target_col=target_col,
                era_col=era_col,
                data_version=data_version,
                startera=test_start_str,
                endera=test_start_str,
            )

            features_list.append(features)
            targets_list.append(targets)
            groups_list.append(groups)
            weights_list.append(weights)

        return (
            pd.concat(features_list),
            pd.concat(targets_list),
            pd.concat(groups_list),
            pd.concat(weights_list),
        )
    else:
        features, targets, groups, weights = load_numerai_data_era(
            data_folder,
            feature_metadata=feature_metadata,
            resample=resample,
            resample_freq=resample_freq,
            target_col=target_col,
            era_col=era_col,
            data_version=data_version,
            startera=startera,
            endera=endera,
        )
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


class FeatureMomentumModel:
    def __init__(
        self,
        lookback=52,
        shift=6,
        correlation_file_path=None,
        portfolio_file_path=None,
        target_col=None,
        seed=0,
    ):
        self.seed = seed
        self.lookback = lookback
        self.shift = shift
        self.correlation_file_path = correlation_file_path
        self.portfolio_file_path = portfolio_file_path
        self.target_col = target_col

    def predict(self, features):
        correlation_matrix = pd.read_parquet(self.correlation_file_path)
        factor_momentum = (
            correlation_matrix.shift(self.shift)
            .fillna(0)
            .rolling(self.lookback)
            .mean()
            .dropna()
        )
        last_momentum = factor_momentum.tail(1).transpose().squeeze()[features.columns]
        preds = features * np.sign(last_momentum)
        return preds.mean(axis=1)

    def copy_performance(self, outputfolder):
        portfolio = pd.read_csv(self.portfolio_file_path, index_col=0)
        portfolio.columns = [f"feature-momentum_None_1_{self.seed}_1-baseline-0"]
        portfolio.to_csv(f"{outputfolder}/feature-momentum_None_1_{self.seed}_1.csv")


def predict_numerai(
    features_raw,
    targets,
    groups,
    trained_model,
    parameters,
    modelname="sample",
    gbm_start_iteration=0,  ## Backward Comptability
    era_col="era",
    debug=False,
):
    ## Score on Dataset

    selected_cols = parameters["parameters"]["model"]["feature_columns"]
    target_col = parameters["parameters"]["model"]["target_columns"]

    ## Transform Features
    if parameters["parameters"]["model"]["feature_engineering"] is not None:
        if parameters["parameters"]["model"]["feature_engineering"] in [
            "numerai",
        ]:
            feature_eng_parameters = parameters["parameters"]["feature_eng"]
            transformer = NumeraiTransformer(**feature_eng_parameters)
            transformer.data = parameters["transformer"]
            features = transformer.transform(
                features_raw[selected_cols], is_train=False
            )
        if parameters["parameters"]["model"]["feature_engineering"] in [
            "numeraiv4",
            "numeraiv4.1",
        ]:
            feature_eng_parameters = parameters["parameters"]["feature_eng"]
            transformer = NumeraiTransformerV4(**feature_eng_parameters)
            transformer.data = parameters["transformer"]
            features = transformer.transform(
                features_raw[selected_cols], is_train=False
            )
    else:
        features = features_raw[selected_cols]

    ## Run Predictions
    ## For tree-based models can run some of the trees only
    if parameters["parameters"]["model"]["tabular_model"] in [
        "lightgbm",
        "lightgbm-gbdt",
        "lightgbm-dart",
        "lightgbm-goss",
    ]:
        ## Backward Compatability
        if "additional" in parameters["parameters"] and gbm_start_iteration is None:
            gbm_start_iteration = parameters["parameters"]["additional"].get(
                "gbm_start_iteration", 0
            )
        start_iteration = min(gbm_start_iteration, int(trained_model.num_trees() * 0.75))
        predictions_raw = trained_model.predict(
            features, start_iteration=start_iteration
        )
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "xgboost",
    ]:
        if hasattr(trained_model, "best_iteration"):
            end_iteration = trained_model.best_iteration
        else:
            end_iteration = trained_model.num_boosted_rounds()
        start_iteration = min(gbm_start_iteration, int(end_iteration * 0.75))
        xgboost_features = xgboost.DMatrix(features)
        predictions_raw = trained_model.predict(
            xgboost_features,
            iteration_range=(start_iteration, end_iteration),
        )
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "Numerai-MLP",
        "Numerai-LSTM",
    ]:
        predictions_raw = trained_model.predict(features.values)
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "tabnet",
    ]:
        predictions_raw = trained_model.predict(features.values)
    elif parameters["parameters"]["model"]["tabular_model"] in [
        "feature-momentum",
    ]:
        trained_model = FeatureMomentumModel(**parameters["parameters"]["tabular"])
        predictions_raw = trained_model.predict(features)
    else:
        ## General Model which implements a predict method
        predictions_raw = trained_model.predict(features)

    ## Process Predictions into DataFrame
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


def predict_numerai_multiple(
    Numerai_Model_Names,
    correlation_matrix=None,
    filename="data/v4_all_int8.parquet",
    data_version="v4",
    startera=None,
    endera=None,
    debug=False,
    era_col="era",
    target_col=["target"],
    embargo=26,
    gbm_start_iteration=0,
):

    features, targets, groups, weights = load_numerai_data(
        filename,
        target_col=target_col,
        era_col=era_col,
        data_version=data_version,
        startera=startera,
        endera=endera,
    )

    INDEX_COL_NAMES = features.index.names

    prediction_df_list = list()
    score_df_list = list()

    for Numerai_Model_Name in Numerai_Model_Names:
        modelname = Numerai_Model_Name.replace(".parameters", ".model")
        parameters = joblib.load(Numerai_Model_Name)
        most_recent_model = load_best_model(
            parameters["parameters"]["model"]["tabular_model"], modelname
        )

        ## Check Embargo Period for Numerai Classic Models
        if data_version in [
            "v4",
            "v4.1",
            "v5",
            "v6",
        ]:
            test_start = shift_era(
                parameters["parameters"]["model"]["validation_end"], embargo
            )
            required_index = groups[groups >= test_start].index
        else:
            required_index = groups.index

        if debug:
            print(modelname, test_start)

        if required_index.shape[0] > 0:

            prediction_df = predict_numerai(
                features.loc[required_index],
                targets.loc[required_index],
                groups.loc[required_index],
                most_recent_model,
                parameters,
                modelname=modelname,
                gbm_start_iteration=gbm_start_iteration,
                era_col=era_col,
                debug=debug,
            )
            prediction_df_list.append(prediction_df)

            if debug:
                print(prediction_df.columns, prediction_df.shape)

    if len(prediction_df_list) > 0:
        output_cols = [era_col, "prediction"] + target_col
        average_prediction_df = (
            pd.concat(prediction_df_list, axis=0)
            .groupby(INDEX_COL_NAMES)[output_cols]
            .mean()
        )
        average_prediction_df[era_col] = groups
        if debug:
            print(average_prediction_df.columns, average_prediction_df.shape)

        return average_prediction_df.sort_values(era_col), prediction_df_list
    else:
        return pd.DataFrame(), pd.DataFrame()


"""
Score Numerai Models with FN using CUDA 

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
        ## Computation on CUDA
        if torch.cuda.is_available():
            temp = (
                scipy.stats.rankdata(df[prediction_col], method="ordinal") - 0.5
            ) / len(df[prediction_col])
            df["normalised_prediction"] = scipy.stats.norm.ppf(temp)
            ## Neutralised targets (FNC)
            if proportion > 0 and len(riskiest_features) > 0:
                exposures = cp.asarray(features.loc[df.index, riskiest_features])
                normalised_prediction = cp.asarray(df["normalised_prediction"])
                gram_mtx = cp.dot(cp.linalg.pinv(exposures), normalised_prediction)
                projected_values = normalised_prediction - cp.asarray(
                    proportion
                ) * cp.dot(exposures, gram_mtx)
                df["neutralised_prediction"] = projected_values.get()
                df["neutralised_prediction"] = (
                    df["neutralised_prediction"] / df["neutralised_prediction"].std()
                )
                output["neutralised_correlation"] = cp.corrcoef(
                    cp.asarray(df[target_col_name]),
                    cp.asarray(df["neutralised_prediction"].rank(pct=True)),
                )[0, 1].get()
                prediction_df.loc[df.index, "neutralised_prediction"] = df[
                    "neutralised_prediction"
                ].rank(pct=True)
            else:
                output["neutralised_correlation"] = cp.corrcoef(
                    cp.asarray(df[target_col_name]),
                    cp.asarray(df[prediction_col].rank(pct=True)),
                )[0, 1].get()
                prediction_df.loc[df.index, "neutralised_prediction"] = df[
                    prediction_col
                ].rank(pct=True)
        ### Computation on CPU
        else:
            ## Normalise prediction
            temp = (
                scipy.stats.rankdata(df[prediction_col], method="ordinal") - 0.5
            ) / len(df[prediction_col])
            df["normalised_prediction"] = scipy.stats.norm.ppf(temp)
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
                output["neutralised_correlation"] = np.corrcoef(
                    df[target_col_name], df[prediction_col].rank(pct=True)
                )[0, 1]
                prediction_df.loc[df.index, "neutralised_prediction"] = df[
                    prediction_col
                ].rank(pct=True)
        correlations_by_era.append(output)
    ## Generate Overall files
    correlations_by_era_all = pd.DataFrame.from_records(correlations_by_era)
    prediction_df["model_name"] = modelname
    correlations_by_era_all["model_name"] = modelname
    return prediction_df, correlations_by_era_all



"""
Linear Factor Model
Factor Timing
rawdata: pd.DataFrame: Numerai dataset with columns containing the 1149 features and 20 targets, index id 
"""


def numerai_feature_correlation_matrix(
    rawdata, feature_col=None, target_col_name=None, era_col="era"
):

    output = dict()
    for i, df in rawdata.groupby(era_col):
        corr_dict = dict()
        for feature in feature_col:
            corr_dict[feature] = np.corrcoef(
                df[feature].fillna(2).astype(float), df[target_col_name]
            )[0, 1]
        output[i] = corr_dict

    return pd.DataFrame.from_records(output).transpose()[feature_col]


def numerai_feature_momentum(
    data_folder="../data/era",
    output_folder="../data/feature_momentum",
    data_version="v4",
    startera="0001",
    endera="1037",
    era_col="era",
    lookback=52,
    update_correlation_mtx=True,
    feature_col=None,
):

    if update_correlation_mtx:
        ## Calculate Correlation Matrix
        for i in range(int(startera), int(endera) + 1):
            if i <= 9:
                test_start_str = "000" + str(i)
            elif i <= 99:
                test_start_str = "00" + str(i)
            elif i <= 999:
                test_start_str = "0" + str(i)
            else:
                test_start_str = str(i)
            data_file = f"{data_folder}/{data_version}_{test_start_str}_int8.parquet"
            rawdata = pd.read_parquet(data_file)
            if feature_col is None:
                feature_col = [x for x in rawdata.columns if x.startswith("feature_")]
            target_cols = [x for x in rawdata.columns if x.startswith("target_")]
            for target_col in target_cols:
                correlation_file = f"{output_folder}/{target_col}_corr.parquet"
                if os.path.exists(correlation_file):
                    correlation_matrix_old = pd.read_parquet(correlation_file)
                    if test_start_str > correlation_matrix_old.index[-1]:
                        rawdata_copy = rawdata.dropna(subset=[target_col]).copy()
                        feature_col = correlation_matrix_old.columns
                        if rawdata_copy.shape[0] > 0:
                            correlation_matrix = numerai_feature_correlation_matrix(
                                rawdata_copy, feature_col, target_col
                            )
                            pd.concat(
                                [correlation_matrix_old, correlation_matrix]
                            ).to_parquet(correlation_file)
                else:
                    rawdata_copy = rawdata.dropna(subset=[target_col]).copy()
                    if rawdata_copy.shape[0] > 0:
                        correlation_matrix = numerai_feature_correlation_matrix(
                            rawdata_copy, feature_col, target_col
                        )
                        correlation_matrix.to_parquet(correlation_file)

    data_file = f"{data_folder}/{data_version}_{endera}_int8.parquet"
    rawdata = pd.read_parquet(data_file)
    # feature_col = [x for x in rawdata.columns if x.startswith("feature_")]
    target_cols = [x for x in rawdata.columns if x.startswith("target_")]

    ## Factor Momentum Portfolio
    for target_col in target_cols:
        correlation_file = f"{output_folder}/{target_col}_corr.parquet"
        correlation_matrix = pd.read_parquet(correlation_file)
        feature_col = correlation_matrix.columns

        if "60" in target_col:
            gap = 14
        else:
            gap = 6

        factor_momentum = (
            correlation_matrix.shift(gap).fillna(0).rolling(lookback).mean().dropna()
        )
        factor_volatility = (
            correlation_matrix.shift(gap).fillna(0).rolling(lookback).std().dropna()
        )
        fm_max_index = factor_momentum.index.max()
        fm_min_index = factor_momentum.index.min()

        factor_momentum_eras = factor_momentum.unstack(level=0).reset_index()
        factor_momentum_eras.columns = ["feature_name", "era", "momentum"]
        factor_volatility_eras = factor_volatility.unstack(level=0).reset_index()
        factor_volatility_eras.columns = ["feature_name", "era", "volatility"]

        for i in range(int(startera), int(endera) + 1):
            if i <= 9:
                test_start_str = "000" + str(i)
            elif i <= 99:
                test_start_str = "00" + str(i)
            elif i <= 999:
                test_start_str = "0" + str(i)
            else:
                test_start_str = str(i)

            if (test_start_str <= fm_max_index) & (test_start_str >= fm_min_index):
                factor_file = f"{output_folder}/{target_col}_feature_momentum.csv"
                if os.path.exists(factor_file):
                    factor_portfolio_old = pd.read_csv(factor_file, index_col=0)
                    factor_portfolio_old.index = pd.to_datetime(
                        factor_portfolio_old.index
                    )
                    if (
                        convert_era_to_datetime(test_start_str)
                        > factor_portfolio_old.index[-1]
                    ):
                        update = True
                    else:
                        update = False
                else:
                    update = True

                if update:
                    ## Read Data
                    data_file = (
                        f"{data_folder}/{data_version}_{test_start_str}_int8.parquet"
                    )
                    df = pd.read_parquet(data_file)
                    feature_col = [x for x in df.columns if x.startswith("feature_")]
                    df[feature_col] = df[feature_col].fillna(2) - 2

                    ## Factor Momentum
                    portfolio_predictions = df[[era_col, "target"]]
                    per_era = df[feature_col] * np.sign(
                        factor_momentum.loc[test_start_str, feature_col]
                    )
                    portfolio_predictions["prediction"] = per_era.mean(axis=1)
                    prediction_era, correlations_era = score_numerai(
                        portfolio_predictions,
                        df[feature_col],
                        None,
                        proportion=0,
                        modelname=f"{target_col}_feature_momentum-baseline-0",
                        target_col_name="target",
                    )
                    factor_porfolio = create_era_index(
                        correlations_era.pivot(
                            index="era",
                            columns=["model_name"],
                            values=["neutralised_correlation"],
                        )
                    )
                    if os.path.exists(factor_file):
                        factor_portfolio_old = pd.read_csv(factor_file, index_col=0)
                        factor_portfolio_old.index = pd.to_datetime(
                            factor_portfolio_old.index
                        )
                        pd.concat(
                            [
                                factor_portfolio_old,
                                factor_porfolio["neutralised_correlation"],
                            ]
                        ).to_csv(factor_file)
                    else:
                        factor_porfolio["neutralised_correlation"].to_csv(factor_file)


"""

Benchmark Performances of Numerai Models 

Run Model Performances for models trained with a single ML model 

"""

def dynamic_feature_neutralisation(
    prediction_df,
    features_raw,
    feature_corr=None,
    features_optimizer=None,
    modelname="sample",
    era_col="era",
    target_col=["target"],
    cutoff=420,
    gap=6,
    lookback=52,
    proportion=1,
    debug=False,
):

    if features_optimizer is None:
        features_optimizer = features_raw.columns[:cutoff]

    if feature_corr is None:
        ## Get index by era
        prediction_dynamic = list()
        correlation_dynamic = list()
        for i, df in prediction_df.groupby(era_col):
            if debug:
                print(modelname, i, df.shape)
            prediction_df_era = prediction_df.loc[df.index]
            features_raw_era = features_raw.loc[df.index]
            ## Baseline
            prediction_df_era_new, correlations_by_era = score_numerai(
                prediction_df_era,
                features_raw_era,
                list(),
                proportion=0,
                modelname=f"{modelname}-baseline",
                target_col_name=target_col[0],
                era_col=era_col,
                debug=debug,
            )
            prediction_dynamic.append(prediction_df_era_new.copy())
            correlation_dynamic.append(correlations_by_era)
        return pd.concat(prediction_dynamic, axis=0), pd.concat(
            correlation_dynamic, axis=0
        )

    else:
        ## Generate Feature Momentum Leaderboard
        factor_mean = (
            feature_corr.shift(gap).fillna(0).rolling(lookback).mean().dropna()
        )
        factor_volatility = (
            feature_corr.shift(gap).fillna(0).rolling(lookback).std().dropna()
        )
        factor_skew = (
            feature_corr.shift(gap).fillna(0).rolling(lookback).skew().dropna()
        )
        factor_kurt = (
            feature_corr.shift(gap).fillna(0).rolling(lookback).kurt().dropna()
        )
        factor_drawdown = (
            (-1 * (feature_corr.cumsum() - feature_corr.cumsum().cummax()).cummin())
            .shift(gap)
            .fillna(0)
        )
        factor_sharpe = factor_mean / factor_volatility
        factor_calmar = factor_mean / factor_drawdown
        factor_autocorrelation = (
            feature_corr.rolling(lookback)
            .corr(feature_corr.shift(4))
            .shift(gap)
            .fillna(0)
        )

        fm_max_index = factor_mean.index.max()
        fm_min_index = factor_mean.index.min()

        ##
        factor_flavour_eras = dict()
        for flavour in [
            "mean",
            "volatility",
        ]:
            factor_flavour_eras[flavour] = (
                locals()[f"factor_{flavour}"].unstack(level=0).reset_index()
            )
            factor_flavour_eras[flavour].columns = ["feature_name", "era", flavour]

        ## Get index by era
        prediction_dynamic = list()
        correlation_dynamic = list()
        for i, df in prediction_df.groupby(era_col):
            if debug:
                print(modelname, i, df.shape)
            if (i <= fm_max_index) & (i >= fm_min_index):
                prediction_df_era = prediction_df.loc[df.index]
                features_raw_era = features_raw.loc[df.index]

                ## Baseline
                prediction_df_era_new, correlations_by_era = score_numerai(
                    prediction_df_era,
                    features_raw_era,
                    list(),
                    proportion=0,
                    modelname=f"{modelname}-baseline",
                    target_col_name=target_col[0],
                    era_col=era_col,
                    debug=debug,
                )
                prediction_dynamic.append(prediction_df_era_new.copy())
                correlation_dynamic.append(correlations_by_era)

                ## For v4-data only
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

                ## Optimizer
                prediction_df_era_new, correlations_by_era = score_numerai(
                    prediction_df_era,
                    features_raw_era,
                    features_optimizer,
                    proportion=proportion,
                    modelname=f"{modelname}-optimizer",
                    target_col_name=target_col[0],
                    era_col=era_col,
                    debug=debug,
                )
                prediction_dynamic.append(prediction_df_era_new.copy())
                correlation_dynamic.append(correlations_by_era)

                ### Dynamic Feature Neutralisation by different criteria
                DFN_params = list()
                for flavour in [
                    "mean",
                    "volatility",
                ]:
                    for direction in [
                        "tail",
                        "head",
                    ]:
                        for size in [
                            420,
                        ]:
                            if direction == "tail":
                                name = f"high_{flavour}_"
                            else:
                                name = f"low_{flavour}_"
                            if size == 420:
                                name = name + "standard"
                            elif size == 105:
                                name = name + "small"
                            temp = (flavour, size, direction, name)
                            DFN_params.append(temp)

                for DFN_param in DFN_params:
                    flavour = DFN_param[0]
                    factor_flavour_era = factor_flavour_eras[flavour][
                        factor_flavour_eras[flavour]["era"] == i
                    ]
                    selected_features = getattr(
                        factor_flavour_era.sort_values(flavour), DFN_param[2]
                    )(DFN_param[1])["feature_name"]

                    selected_features = list(set(selected_features) - set(bad_features))

                    prediction_df_era_new, correlations_by_era = score_numerai(
                        prediction_df_era,
                        features_raw_era,
                        selected_features,
                        proportion=proportion,
                        modelname=f"{modelname}-{DFN_param[3]}",
                        target_col_name=target_col[0],
                        era_col=era_col,
                        debug=debug,
                    )
                    prediction_dynamic.append(prediction_df_era_new.copy())
                    correlation_dynamic.append(correlations_by_era)

        return pd.concat(prediction_dynamic, axis=0), pd.concat(
            correlation_dynamic, axis=0
        )


def save_model_performance_test(
    Numerai_Model_Names,
    feature_corr,
    features_optimizer,
    startera=None,
    endera=None,
    data_file="data/v4_all_int8.parquet",
    data_version="v4",
    target_col=["target"],
    debug=False,
    gbm_start_iteration=0,
):

    (average_prediction_df, prediction_df_list,) = predict_numerai_multiple(
        Numerai_Model_Names,
        feature_corr,
        filename=data_file,
        data_version=data_version,
        startera=startera,
        endera=endera,
        debug=debug,
        target_col=target_col,
        gbm_start_iteration=gbm_start_iteration,
    )

    del prediction_df_list

    MODEL_NAME = Numerai_Model_Names[0].split(".parameters")[0].split("/")[-1]
    MODEL_NAME = MODEL_NAME + f"_{len(Numerai_Model_Names)}"

    (features, targets, groups, weights,) = load_numerai_data(
        data_file,
        resample_freq=1,
        startera=startera,
        endera=endera,
        target_col=target_col,
        data_version=data_version,
    )

    dynamic_predictions, dynamic_correlations = dynamic_feature_neutralisation(
        average_prediction_df,
        features,
        feature_corr,
        features_optimizer,
        target_col=target_col,
        modelname=MODEL_NAME,
        debug=debug,
    )
    summary_correlations = dynamic_correlations.pivot(
        index="era", columns="model_name", values=["neutralised_correlation"]
    ).dropna()
    strategy_flavour = pd.DataFrame.from_records(
        summary_correlations.apply(strategy_metrics, axis=0),
        index=summary_correlations.columns,
    )
    if data_version == "signals":
        return (
            strategy_flavour,
            summary_correlations,
            dynamic_predictions,
        )
    else:
        return (
            strategy_flavour,
            create_era_index(summary_correlations),
            dynamic_predictions,
        )


## Run Numerai Model Performances for both Classic and Signals tournament
def run_numerai_models_performances(
    Numerai_Model_Names,
    feature_corr,
    features_optimizer,
    PERFORMANCES_FOLDER,
    data_file="data/v4_all_int8.parquet",
    data_version="v4",
    target_col=["target"],
    gbm_start_iteration=0,
):

    ## Calculate Starting Era
    parametername = Numerai_Model_Names[0]
    no_models = len(Numerai_Model_Names)
    stem = parametername.split("/")[-1].replace(".parameters", "")
    correlations_filename = f"{PERFORMANCES_FOLDER}/{stem}_{no_models}.csv"
    if os.path.exists(parametername):
        parameters = joblib.load(parametername)
        if data_version == "signals":
            test_start = parameters["parameters"]["model"]["validation_end"]
            test_end = datetime.datetime.strptime("2099-12-31", "%Y-%m-%d")
        else:
            test_start = shift_era(
                parameters["parameters"]["model"]["validation_end"], gap=14
            )
            test_end = feature_corr.index[-1]
        if os.path.exists(correlations_filename):
            most_recent_date = pd.read_csv(correlations_filename, index_col=0).index[-1]
            if data_version == "signals":
                test_start = datetime.datetime.strptime(most_recent_date, "%Y-%m-%d")
            else:
                test_start = shift_era(convert_datetime_to_era(most_recent_date), gap=1)
        print(f"Model Performances {test_start} {test_end}")
        ### Get Model Predictions for the latest eras
        if test_end >= test_start:
            (
                validate_performance,
                validate_correlations,
                validate_predictions,
            ) = save_model_performance_test(
                Numerai_Model_Names,
                feature_corr,
                features_optimizer,
                startera=test_start,
                endera=test_end,
                data_file=data_file,
                data_version=data_version,
                target_col=target_col,
                gbm_start_iteration=gbm_start_iteration,
            )
            ## Update Model Performances
            output = validate_correlations["neutralised_correlation"]
            if os.path.exists(correlations_filename):
                old_file = pd.read_csv(correlations_filename, index_col=0)
                df = pd.concat([old_file, output.dropna()])
                df.index = pd.to_datetime(df.index)
                df[~df.index.duplicated()].sort_index().to_csv(correlations_filename)
            else:
                output.dropna().to_csv(correlations_filename)
