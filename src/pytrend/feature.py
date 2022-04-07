#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of feature enginnering methods for time-series data
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

from .util import align_features_target, RollingTSTransformer, GroupedTimeSeriesSplit

import pandas as pd
import numpy as np
import joblib

## Sklearn Transformers
from sklearn.base import TransformerMixin, BaseEstimator

## Feature Engineering packages
import iisignature as iis
from sklearn.decomposition import PCA


### Feature Union that persists transformer history for online learning and supports multi-processing
from joblib import Parallel, delayed


def _transform_one(transformer, X, y, **fit_params):
    res = transformer.transform(X)
    return res


class FeatureUnionOnline:
    def __init__(
        self,
        transformers,
        n_jobs=None,
        verbose=None,
    ):
        self.transformers = transformers
        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_transform_one)(
                trans,
                X,
                None,
            )
            for trans in self.transformers
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return pd.concat(Xs, axis=1)


### Supervised Feature Engineering methods


#### Numerai Transformer to augment the Gaussian normalised features
###
### X: pd.DataFrame (id x features)
###
class NumeraiTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        usesquare=False,
        seed=0,
        no_product_features=0,
        no_pca_features=0,
        dropout_pct=0,
    ):
        self.usesquare = usesquare
        self.no_product_features = no_product_features
        self.no_pca_features = no_pca_features
        self.seed = seed
        self.dropout_pct = dropout_pct

    ## Transform Numerai Features with mean zero (-2,-1,0,1,2)
    def transform(self, X, is_train=True):

        ## Numpy Random Number Generator
        rng = np.random.default_rng(self.seed)

        ## Create Default Arguments
        if not hasattr(self, "usesquare"):
            self.usesquare = False
        if not hasattr(self, "no_product_features"):
            self.no_product_features = 0
        if not hasattr(self, "dropout_pct"):
            self.dropout_pct = 0
        if not hasattr(self, "seed"):
            self.seed = 0
        if not hasattr(self, "no_pca_features"):
            self.no_pca_features = 0

        if self.usesquare:
            squareX = np.square(X)
            squareX.columns = ["{}_square".format(x) for x in X.columns]
        else:
            squareX = pd.DataFrame()

        ## Pair Transforms
        if self.no_product_features > 0:
            if is_train:
                col1 = rng.choice(X.columns, self.no_product_features)
                col2 = rng.choice(
                    X.columns,
                    self.no_product_features,
                )
                self.product_features = pd.DataFrame(
                    {
                        "col1": col1,
                        "col2": col2,
                    }
                ).drop_duplicates()
            productX = (
                X[self.product_features["col1"]]
                * X[self.product_features["col2"]].values
            )
            productX.columns = [
                f"feature_product_{i}" for i in range(self.product_features.shape[0])
            ]
        else:
            productX = pd.DataFrame()

        if self.no_pca_features > 0:
            if is_train:
                self.pca_transformer = PCA(
                    n_components=self.no_pca_features, random_state=self.seed
                )
                self.pca_transformer.fit(X)
                pcaX = pd.DataFrame(self.pca_transformer.transform(X), index=X.index)
            else:
                pcaX = pd.DataFrame(self.pca_transformer.transform(X), index=X.index)
            pcaX.columns = [f"feature_pca_{i}" for i in range(self.no_pca_features)]
        else:
            pcaX = pd.DataFrame()

        ## Concat All Features to output
        transformed_features = pd.concat(
            [
                X.astype(np.int8),
                squareX.astype(np.int8),
                productX.astype(np.int8),
                pcaX,
            ],
            axis=1,
        )

        ## Dropout Matrix
        if self.dropout_pct > 0 and is_train:
            dropout_matrix = 1 - rng.binomial(
                1, self.dropout_pct, transformed_features.shape
            )
            transformed_features = transformed_features * dropout_matrix.astype(np.int8)
        return transformed_features


###
### Benchmark Feature Engineering methods
###
### For a given cross validation, feature engineering method, tabular model and hyper-parameters and metric
###
### features: pd.DataFrame (timestamp x n_features)
### target: pd.DataFrame  (timestamp x n_targets) The first column is the main target with the rest for ensemble regularisation
### n_splits: int (How many time-series cv split)
### max_train_size: int (Longest time-series cv split)
### gap: int (Gap between train and test)
### feature_eng: str (tsfresh/signature)
### feature_eng_parameters: dict (parameters for the feature engineering method)
### tabular_model: str (xgboost/lightbgm/catboost/tabnet)
### tabular_hyper: dict (hyper-parameters for the tabular model)
### interval: int (forward return period to normalise sharpe ratio)
###


def benchmark_features_transform(
    X_train,
    y_train,
    X_test=None,
    group_train=None,
    group_test=None,
    feature_eng="numerai",
    feature_eng_parameters=None,
    n_jobs=20,
    debug=False,
):

    ### Numerai Transforms (z-scores or quantile-transformed features)
    if feature_eng == "numerai":
        if feature_eng_parameters is None:
            feature_eng_parameters = {
                "usesquare": False,
                "no_product_features": 0,
                "seed": 10,
            }
        transformer = NumeraiTransformer(**feature_eng_parameters)

    if feature_eng is not None:
        extracted_features_train = pd.DataFrame(
            transformer.transform(X_train), index=X_train.index
        )
        if X_test is not None:
            extracted_features_test = pd.DataFrame(
                transformer.transform(X_test, is_train=False), index=X_test.index
            )
        else:
            extracted_features_test = None
        return transformer, extracted_features_train, extracted_features_test
    else:
        if X_test is not None:
            return None, X_train, X_test
        else:
            return None, X_train, None


#### Rolling Summary
####
### For a given lookback, create summary statistics for a price series or any I(1) integrated time-series
###
class RollingSummaryTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        lookback,
        log_change=False,
        agg_cols=None,
    ):
        self.lookback = lookback
        self.log_change = log_change
        self.agg_cols = agg_cols
        self.history = None

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.history is not None:
            X = pd.concat([self.history, X], axis=0)
        if not self.agg_cols:
            self.agg_cols = [
                "mean",
                "std",
                "skew",
                "kurt",
                "max",
                "min",
            ]

        if not self.log_change:
            output = X.rolling(self.lookback, min_periods=1).agg(self.agg_cols)
        else:
            logX = logX = np.log(X) - np.log(X).shift(1)
            output = logX.rolling(self.lookback, min_periods=1).agg(self.agg_cols)

        output.columns = [
            "lookback_{}_summary_{}".format(self.lookback, i)
            for i in range(output.shape[1])
        ]

        if self.history is not None:
            history_length = self.history.shape[0]
            self.history = X.iloc[-1 * self.lookback :, :]
            output_length = X.shape[0] - self.lookback
            return output.iloc[-1 * output_length :, :]
        else:
            self.history = X.iloc[-1 * self.lookback :, :]
            return output


#### iisignature
####
#### Compute rolling window signatures with normaliation
####
class RollingWindowSignature:
    def __init__(self, row_dimension, rolling_window_length, sig_level):

        self.d = row_dimension
        self.num_sig_rows = rolling_window_length - 1
        self.sig_level = sig_level
        self.sig_length = iis.siglength(self.d, self.sig_level)
        self.nan_row = np.empty((self.sig_length,))
        self.nan_row.fill(np.nan)

        levels = [self.d] + [
            iis.siglength(self.d, x) - iis.siglength(self.d, x - 1)
            for x in range(2, sig_level + 1)
        ]

        self.reverse_path_map = []
        for l in [[-1 if l % 2 == 0 else 1] * levels[l] for l in range(len(levels))]:
            self.reverse_path_map.extend(l)
        self.reverse_path_map = np.array(self.reverse_path_map)

        self.n = 0
        self.initial_fill = True
        self.last_row = None
        self.hist = np.zeros((self.num_sig_rows + 1, self.d))
        self.signatures = np.zeros((self.num_sig_rows, self.sig_length))
        self.signatures_total_product = np.zeros(
            self.sig_length,
        )

    ## Iterate through
    def get_last_signature(self, row):

        if self.last_row is None:
            # No rows yet
            self.last_row = row
            self.hist[self.n, :] = row
            self.n += 1
            return self.nan_row

        elif self.initial_fill:
            # Filling the signatures with the two step paths
            self.signatures[self.n - 1] = iis.sig(
                np.array([self.last_row, row]), self.sig_level
            )
            self.signatures_total_product = iis.sigcombine(
                self.signatures_total_product,
                self.signatures[self.n - 1],
                self.d,
                self.sig_level,
            )
            # Update cached history
            self.last_row = row
            self.hist[self.n % (self.num_sig_rows + 1), :] = row
            self.n += 1

            if self.n == self.num_sig_rows + 1:
                self.initial_fill = False
                scales = 1 / np.std(self.hist, axis=0)
                return iis.sigscale(
                    self.signatures_total_product, scales, self.sig_level
                )
            else:
                return self.nan_row
        else:
            # Reverse out the inital step, and add the new two step path
            replace_row = (self.n - 1) % self.num_sig_rows
            # Remove and Update rows to existing singatures
            self.signatures_total_product = iis.sigcombine(
                self.signatures[replace_row] * self.reverse_path_map,
                self.signatures_total_product,
                self.d,
                self.sig_level,
            )
            self.signatures[replace_row] = iis.sig(
                np.array([self.last_row, row]), self.sig_level
            )
            self.signatures_total_product = iis.sigcombine(
                self.signatures_total_product,
                self.signatures[replace_row],
                self.d,
                self.sig_level,
            )
            # Update cached history
            self.last_row = row
            self.hist[self.n % (self.num_sig_rows + 1), :] = row
            scales = 1 / np.std(self.hist, axis=0)
            self.n += 1

            return iis.sigscale(self.signatures_total_product, scales, self.sig_level)

    def get_all_signature(self, X):
        return pd.DataFrame(
            [self.get_last_signature(X.iloc[r].to_numpy()) for r in range(X.shape[0])]
        )


class SignatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        no_channels,
        lookback,
        signature_level,
    ):
        self.lookback = lookback
        self.signature_level = signature_level
        self.no_channels = no_channels
        self.signature = RollingWindowSignature(no_channels, lookback, signature_level)

    ## Can also be used to transform data in an online fashion by transform
    def transform(self, X):
        index = X.index.copy()
        transformed_signature = self.signature.get_all_signature(X)
        transformed_signature.index = index
        transformed_signature.columns = [
            "lookback_{}_signature_{}".format(self.lookback, i)
            for i in range(transformed_signature.shape[1])
        ]
        return transformed_signature


class PriceDataTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        no_channels,
        lookback,
        signature_level,
    ):
        self.lookback = lookback
        self.no_channels = no_channels
        self.signature_level = signature_level
        self.summarytransformer = RollingSummaryTransformer(lookback, log_change=True)
        self.signaturetransformer = SignatureTransformer(
            no_channels, lookback, signature_level
        )

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        summaryX = self.summarytransformer.transform(X)
        signatureX = self.signaturetransformer.transform(X)
        output = pd.concat(
            [
                summaryX,
                signatureX,
            ],
            axis=1,
        )
        return output
