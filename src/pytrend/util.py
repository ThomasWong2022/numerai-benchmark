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
import joblib
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


### Strategy metrics
##
## strategy: pd.Series (timestamp x ), usually to be daily return of a trading strategy
##
def strategy_metrics(strategy, interval=1, numerai=True):
    results = dict()
    if strategy.std() > 0:
        results["sharpe"] = strategy.mean() / strategy.std()
    else:
        results["sharpe"] = np.nan
    results["mean"] = strategy.mean()
    results["volatility"] = strategy.std()
    results["skew"] = strategy.skew()
    results["kurtosis"] = strategy.kurtosis()
    if numerai:
        portfolio = strategy.cumsum()
    else:
        portfolio = (1 + strategy).cumprod()
    if numerai:    
        dd = (portfolio - portfolio.cummax())
    else:
        dd = (portfolio - portfolio.cummax()) / portfolio.cummax()
    results["max_drawdown"] = -1 * dd.cummin().min()
    if results["max_drawdown"] > 0:
        results["RMDD"] = results["mean"] / results["max_drawdown"]
    else:
        results["RMDD"] = np.nan
    return results


### Data Pre-processing
##
##
## Align features and target to valid index
##
## features: pd.DataFrame/pd.Series
## target: pd.DataFrame/pd.Series
##
def align_features_target(features, target, large_value=1e6):
    ## Flatten multi-index column names for tsfresh
    if isinstance(features, pd.DataFrame):
        if features.columns.nlevels > 1:
            features.columns = [
                "_".join(column).rstrip("_")
                for column in features.columns.to_flat_index()
            ]
    ## Remove rows with na and align features and target to same length
    features.replace(np.inf, large_value, inplace=True)
    features.replace(-np.inf, -1 * large_value, inplace=True)
    features = features.dropna()
    target = target.dropna()
    valid_index = features.index.intersection(target.index)
    features = features.reindex(valid_index)
    target = target.reindex(valid_index)
    return features, target


### Cross-Validation Schemes

### TimeSeries Cross Validation Grouped
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
                if self.debug:
                    print(train_splits[i], test_splits[i])
                yield (
                    groups[groups.isin(train_splits[i])].index,
                    groups[groups.isin(test_splits[i])].index,
                )


#### Data Dimension Transformer
####
#### Currently Implemeted: Constant lookback size with zero-padding
####
### Convert from 2D DataFrame, given a lookback size into nested DataFrames for sktime transformers


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
