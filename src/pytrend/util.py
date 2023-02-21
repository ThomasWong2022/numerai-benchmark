#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of tools for data pre-processing for non-stationary time-series and tabular data
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
import joblib, os, glob

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


"""
Strategy Metrics , Regime Analysis 
"""


def strategy_metrics(strategy, interval=1, numerai=True, accuracy=4):
    results = dict()
    results["mean"] = np.around(strategy.mean(), accuracy)
    results["volatility"] = np.around(strategy.std(), accuracy)
    results["skew"] = np.around(strategy.skew(), accuracy)
    results["kurtosis"] = np.around(strategy.kurtosis(), accuracy)
    if numerai:
        portfolio = strategy.cumsum()
    else:
        portfolio = (1 + strategy).cumprod()
    if numerai:
        dd = portfolio - portfolio.cummax()
    else:
        dd = (portfolio - portfolio.cummax()) / portfolio.cummax()
    results["max_drawdown"] = np.around(-1 * dd.cummin().min(), accuracy).item()
    if strategy.std() > 0:
        results["sharpe"] = np.around(strategy.mean() / strategy.std(), accuracy)
    else:
        results["sharpe"] = np.around(results["mean"] / 1e-4, accuracy)
    if results["max_drawdown"] > 0:
        results["calmar"] = np.around(
            results["mean"] / results["max_drawdown"], accuracy
        )
    else:
        results["calmar"] = np.around(results["mean"] / 1e-4, accuracy)
    return results


def regime_analysis(
    df,
    performance_col="correlation",
    regime_columns="regime",
):
    ans = df.groupby(regime_columns).agg({performance_col: strategy_metrics})
    ans_df = pd.DataFrame(ans[performance_col].values.tolist())
    ans_df.index = ans.index
    return ans_df.reset_index()


"""
Dynamic Model Selection 
"""


def dynamic_model_selection_masks(performances, gap=6, lookback=52, top_models=1):

    mean = performances.shift(gap).rolling(lookback).mean()
    volatility = performances.shift(gap).rolling(lookback).std()
    skew = performances.shift(gap).rolling(lookback).skew()
    kurt = performances.shift(gap).rolling(lookback).kurt()
    drawdown = (
        -1
        * (
            performances.shift(gap).cumsum() - performances.shift(gap).cumsum().cummax()
        ).cummin()
    )
    sharpe = mean / volatility
    calmar = mean / drawdown

    metric_masks = dict()
    for metric in [
        "mean",
        "volatility",
        "skew",
        "kurt",
        "drawdown",
        "sharpe",
        "calmar",
    ]:
        metric_masks[f"{metric}_min"] = np.where(
            locals()[metric].rank(
                axis=1,
                ascending=True,
                na_option="bottom",
            )
            <= top_models,
            1 / top_models,
            np.nan,
        )
        metric_masks[f"{metric}_max"] = np.where(
            locals()[metric].rank(
                axis=1,
                ascending=False,
                na_option="bottom",
            )
            <= top_models,
            1 / top_models,
            np.nan,
        )

    masks_dataframes = dict()
    for metric in [
        "mean",
        "volatility",
        "skew",
        "kurt",
        "drawdown",
        "sharpe",
        "calmar",
    ]:
        masks_dataframes[f"{metric}_min"] = pd.DataFrame(
            metric_masks[f"{metric}_min"],
            columns=locals()[metric].columns,
            index=locals()[metric].index,
        )
        masks_dataframes[f"{metric}_max"] = pd.DataFrame(
            metric_masks[f"{metric}_max"],
            columns=locals()[metric].columns,
            index=locals()[metric].index,
        )
    return masks_dataframes


def walk_forward_dynamic_models(df_list):

    Model_Sets = dict()
    Imputed_Models = dict()

    for key in [
        "Ensemble",
        "Baseline",
        "Optimizer",
        "Small",
        "Medium",
        "Standard",
        "Average",
    ]:
        Model_Sets[key] = list()

    for dynamic_models in df_list:
        Model_Sets["Ensemble"].append(
            dynamic_models[
                [
                    x
                    for x in dynamic_models.columns
                    if "baseline" in x
                    or "optimizer" in x
                    or ("standard" in x and not "average" in x and not "random" in x)
                ]
            ]
        )
        Model_Sets["Average"].append(
            dynamic_models[
                [
                    x
                    for x in dynamic_models.columns
                    if "baseline" in x or "optimizer" in x or "average" in x
                ]
            ]
        )
        Model_Sets["Baseline"].append(
            dynamic_models[[x for x in dynamic_models.columns if "baseline" in x]]
        )
        Model_Sets["Optimizer"].append(
            dynamic_models[[x for x in dynamic_models.columns if "optimizer" in x]]
        )
        Model_Sets["Standard"].append(
            dynamic_models[
                [
                    x
                    for x in dynamic_models.columns
                    if "standard" in x and not "average" in x and not "random" in x
                ]
            ]
        )
        Model_Sets["Small"].append(
            dynamic_models[
                [
                    x
                    for x in dynamic_models.columns
                    if "small" in x and not "average" in x and not "random" in x
                ]
            ]
        )

    for key in [
        "Ensemble",
        "Baseline",
        "Optimizer",
        "Small",
        "Standard",
        "Average",
    ]:

        models_over_time = pd.concat(Model_Sets[key], axis=1)
        # models_over_time = (
        # models_over_time.transpose()
        # .fillna(models_over_time.mean(axis=1))
        # .transpose()
        # )
        models_over_time = models_over_time.transpose().fillna(0).transpose()
        Imputed_Models[key] = models_over_time.sort_index()

    return Imputed_Models


### Compare Against All Trained Models


def create_leaderboard(
    performances_folder,
    searchkey="*",
    lookback=52,
    no_tops=1,
    model_no_lower=0,
    model_no_upper=1e8,
):

    ## Load csv files
    performances_files = sorted(glob.glob(f"{performances_folder}/{searchkey}.csv"))
    models_list = list()
    for f in performances_files:
        model_no = int(f.split(".csv")[0].split("_")[-2])
        model_seq = int(f.split(".csv")[0].split("_")[-1])
        model_name = "_".join(f.split(".csv")[0].split("/")[-1].split("_")[:3])
        if (
            os.path.isfile(f)
            and model_no_lower <= model_no
            and model_no <= model_no_upper
        ):
            df = pd.read_csv(f, index_col=0).sort_index()
            df = df[~df.index.duplicated()]
            df.index = pd.to_datetime(df.index)
            models_list.append(df)

    dynamic_models_collection = walk_forward_dynamic_models(models_list)

    ### Compute Performances of Portfolios of dynamically selected models
    recent_results = list()
    dynamic_portfolios = dict()
    gap = 6
    criteria = [
        "mean",
        # "calmar",
        # "sharpe",
    ]

    for Sets in [
        "Baseline",
        "Optimizer",
        "Ensemble",
        "Small",
        "Standard",
    ]:
        df = dynamic_models_collection[Sets].sort_index()
        if df.shape[0] > 0:
            dynamic_masks = dynamic_model_selection_masks(
                df, top_models=no_tops, lookback=lookback, gap=gap
            )
            for base_method in criteria:
                for method in [
                    f"{base_method}_max",
                ]:
                    portfolio = (dynamic_masks[method] * df).sum(axis=1, min_count=1)
                    dynamic_portfolios[
                        f"{Sets}_{method}_{no_tops}_lookback_{lookback}"
                    ] = portfolio.tail(df.shape[0] - lookback - gap)
                    performances = strategy_metrics(
                        portfolio.tail(df.shape[0] - lookback - gap)
                    )
                    performances["method"] = method
                    performances["no_tops"] = no_tops
                    performances["sets"] = Sets
                    performances["lookback"] = lookback
                    recent_results.append(performances)

    dynamic_performances = pd.DataFrame(recent_results).dropna()

    leaderboards = dict()
    ## Recent Leaderboards to be used in Model Submissions
    for model_subset in [
        "Baseline",
        "Ensemble",
        "Optimizer",
        "Small",
        "Standard",
    ]:
        leaderboard = pd.DataFrame(
            dynamic_models_collection[model_subset]
            .sort_index()
            .iloc[-1 * lookback :]
            .apply(strategy_metrics)
            .to_dict()
        ).transpose()

        if len(dynamic_models_collection[model_subset].columns) > 0:
            leaderboard.index = dynamic_models_collection[model_subset].columns
            leaderboard["proportion"] = [
                float(x[-1]) for x in leaderboard.index.str.split("-")
            ]
            leaderboard["flavour"] = [x[-2] for x in leaderboard.index.str.split("-")]
            leaderboard["model_seq"] = [
                int("-".join(x[:-2]).split("_")[-1])
                for x in leaderboard.index.str.split("-")
            ]
            leaderboard["model_seed"] = [
                int("-".join(x[:-2]).split("_")[-2])
                for x in leaderboard.index.str.split("-")
            ]
            leaderboard["model_cv"] = [
                "-".join(x[:-2]).split("_")[-3]
                for x in leaderboard.index.str.split("-")
            ]
            leaderboard["model_feature_engineering"] = [
                "-".join(x[:-2]).split("_")[-4]
                for x in leaderboard.index.str.split("-")
            ]
            leaderboard["model_tabular_method"] = [
                "-".join(x[:-2]).split("_")[-5]
                for x in leaderboard.index.str.split("-")
            ]
            leaderboards[model_subset] = leaderboard

            ## Leaderboard Since beginning of data
            if dynamic_models_collection[model_subset].shape[0] < lookback + gap:
                start_of_data = 0
            else:
                start_of_data = lookback + gap
            leaderboard = pd.DataFrame(
                dynamic_models_collection[model_subset]
                .sort_index()
                .iloc[start_of_data:]
                .apply(strategy_metrics)
                .to_dict()
            ).transpose()
            leaderboard.index = dynamic_models_collection[model_subset].columns
            leaderboards[f"{model_subset}-All"] = leaderboard

    return (
        dynamic_performances,
        dynamic_portfolios,
        dynamic_models_collection,
        leaderboards,
    )


"""
Cross Validation Schemes

TimeSeries Grouped CV 

"""


class GroupedTimeSeriesSplit(TimeSeriesSplit):
    def __init__(
        self,
        n_splits=5,
        valid_splits=1,
        max_train_size=None,
        test_size=52 * 2,
        gap=52,
        debug=False,
    ):
        self.n_splits = n_splits
        self.valid_splits = valid_splits
        self.shuffle = False
        self.random_state = None
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.debug = debug

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : pd.Series of shape (n_samples,)
            Group Labels of training data
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        if groups is None:
            # n_samples = X.shape[0]
            n_splits = self.n_splits
            valid_splits = self.valid_splits
            n_folds = n_splits + 1
            gap = self.gap
            test_size = (
                self.test_size if self.test_size is not None else n_samples // n_folds
            )

            # Make sure we have enough samples for the given split parameters
            if n_folds > n_samples:
                raise ValueError(
                    f"Cannot have number of folds={n_folds} greater"
                    f" than the number of samples={n_samples}."
                )
            if n_samples - gap - (test_size * n_splits) <= 0:
                raise ValueError(
                    f"Too many splits={n_splits} for number of samples"
                    f"={n_samples} with test_size={test_size} and gap={gap}."
                )

            indices = X.index
            test_starts = range(
                n_samples - valid_splits * test_size, n_samples, test_size
            )

            for test_start in test_starts:
                train_end = test_start - gap
                if self.max_train_size and self.max_train_size < train_end:
                    yield (
                        indices[max(train_end - self.max_train_size, 0) : train_end],
                        indices[test_start : test_start + test_size],
                    )
                else:
                    yield (
                        indices[:train_end],
                        indices[test_start : test_start + test_size],
                    )
        else:
            ## Get unique groups
            unique_groups = groups.unique()
            gap = self.gap
            ## Calculate test size if not provided
            if self.test_size:
                n_folds = (len(unique_groups) - gap) // self.test_size
            else:
                n_folds = self.n_splits + 1
                self.test_size = len(unique_groups) // n_folds
            test_splits = [
                unique_groups[
                    len(unique_groups)
                    - (i + 1) * self.test_size : len(unique_groups)
                    - i * self.test_size
                ]
                for i in range(n_folds - 1)
            ]
            if self.max_train_size:
                train_splits = [
                    unique_groups[
                        max(
                            len(unique_groups)
                            - (i + 1) * self.test_size
                            - gap
                            - self.max_train_size,
                            0,
                        ) : len(unique_groups)
                        - (i + 1) * self.test_size
                        - gap
                    ]
                    for i in range(n_folds - 1)
                ]
            else:
                train_splits = [
                    unique_groups[: len(unique_groups) - (i + 1) * self.test_size - gap]
                    for i in range(n_folds - 1)
                ]
            for i in range(0, self.valid_splits):
                yield (
                    groups[groups.isin(train_splits[i])].index,
                    groups[groups.isin(test_splits[i])].index,
                )


"""
Data Dimension Transformer
Currently Implemeted: Constant lookback size with zero-padding
Convert from 2D DataFrame, 
given a lookback size into nested DataFrames for sktime transformers
"""


def forward_fill_zero(series, length):
    fill_length = length - series.shape[0]
    fill_series = pd.Series(np.zeros(fill_length))
    return pd.concat([fill_series, series], axis=0).reset_index(drop=True)


### Create rolling windows of nested dataframe for sktime, forward fill zero if there are not enough data at the start
def roll_2D_to_nested(X, lookback=20, normalise=True):
    ## Python index start at zero
    lookback = lookback - 1
    index = X.index
    columns = X.columns
    output = np.empty((len(index), len(columns)), dtype=object)
    for i in range(X.shape[0]):
        for j, c in enumerate(X.columns):
            start_index = max(0, i - lookback)
            recent_rawdata = pd.Series(X.loc[X.index[start_index : i + 1], c])
            if normalise and i >= 1:
                normalised_rawdata = (
                    recent_rawdata - recent_rawdata.mean()
                ) / recent_rawdata.std()
                output[i, j] = forward_fill_zero(normalised_rawdata, lookback + 1)
            else:
                output[i, j] = forward_fill_zero(recent_rawdata, lookback + 1)
    return pd.DataFrame(output, index=index, columns=columns)


class RollingTSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lookback=20, normalise=True):
        self.lookback = lookback
        self.normalise = normalise

    def fit(self, X, y):
        return self

    def transform(self, X):
        output = roll_2D_to_nested(X, self.lookback, self.normalise)
        return output


### Data Pre-processing


def align_features_target(features, target, large_value=1e6):
    ## Flatten multi-index column names for tsfresh
    if isinstance(features, pd.DataFrame):
        if features.columns.nlevels > 1:
            features.columns = [
                "_".join(column).rstrip("_")
                for column in features.columns.to_flat_index()
            ]
    ## Remove rows with na and align features and target to same length
    ##features.replace(np.inf, large_value, inplace=True)
    ##features.replace(-np.inf, -1 * large_value, inplace=True)
    ##features = features.dropna()
    ##target = target.dropna()
    valid_index = features.index.intersection(target.index)
    features = features.reindex(valid_index)
    target = target.reindex(valid_index)
    return features, target
