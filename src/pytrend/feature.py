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


from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import cupy as cp
import cuml
import torch, signatory


import logging

logger = logging.getLogger("Numerai")

from .constant import FEATURE_SETS_V4


"""
Feature Engineering used in Numerai Thesis 
"""


class NumeraiTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        seed=0,
        usesquare=False,
        dropout_pct=0.05,
        no_product_features=10,
        no_pca_features=0,
    ):
        self.seed = seed
        self.usesquare = usesquare
        self.dropout_pct = dropout_pct
        self.no_product_features = no_product_features
        self.no_pca_features = no_pca_features
        ## Data Dictionary to reconsturct transformer during inference
        self.data = dict()

    ## Transform Numerai Features with mean zero (-2,-1,0,1,2)
    def transform(self, X, is_train=True):

        ## Numpy Random Number Generator
        rng = np.random.default_rng(self.seed)

        ## Drop Out Matrix
        if self.dropout_pct > 0 and is_train:
            dropout_matrix = 1 - np.random.binomial(1, self.dropout_pct, X.shape)
            X_val = X.values * dropout_matrix

        if self.usesquare:
            squareX = pd.DataFrame(np.square(X_val), index=X.index)
            squareX.columns = ["{}_square".format(x) for x in X.columns]
        else:
            squareX = pd.DataFrame()

        ## Pair Transforms
        if self.no_product_features > 0:
            if is_train:
                col1 = np.random.choice(X.columns, self.no_product_features)
                col2 = np.random.choice(
                    X.columns,
                    self.no_product_features,
                )
                self.product_features = pd.DataFrame(
                    {
                        "col1": col1,
                        "col2": col2,
                    }
                ).drop_duplicates()
                self.data["product_features"] = self.product_features
            else:
                self.product_features = self.data["product_features"]

            productX = pd.DataFrame(
                np.array(X[self.product_features["col1"]])
                * np.array(X[self.product_features["col2"]]),
                index=X.index,
            )
            productX.columns = [
                f"feature_product_{i}" for i in range(self.product_features.shape[0])
            ]
        else:
            productX = pd.DataFrame()

        ## Concat All Features to output
        transformed_features = pd.concat(
            [
                X.astype(np.int8),
                squareX.astype(np.int8),
                productX.astype(np.int8),
            ],
            axis=1,
        )

        return transformed_features


class SignatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        lookback,
        signature_level,
    ):
        self.lookback = lookback
        self.signature_level = signature_level

    ## Can also be used to transform data in an online fashion by transform
    def transform(self, X):
        history_length = X.shape[0]
        path_class = signatory.Path(
            torch.Tensor(cp.asarray([X.values])), self.signature_level
        )
        sigs = list()
        for i in range(self.lookback, history_length):
            sigs.append(path_class.logsignature(i - self.lookback, i))
        all_sig = torch.concat(sigs)
        transformed_signature = pd.DataFrame(
            all_sig.numpy(), index=X.index[self.lookback :]
        )
        transformed_signature.columns = [
            "lookback_{}_signature_{}".format(self.lookback, i)
            for i in range(transformed_signature.shape[1])
        ]
        return transformed_signature


def features_transform_batch(transformer, data, is_train=True):
    BATCH_SIZE = 10000000000
    start_index = 0
    transformed_features_batches = list()

    while start_index < data.shape[0]:
        data_batch = data.iloc[start_index : start_index + BATCH_SIZE]
        transformed_featrues_batch = pd.DataFrame(
            transformer.transform(data_batch, is_train=is_train), index=data_batch.index
        )
        transformed_features_batches.append(transformed_featrues_batch)
        start_index = start_index + BATCH_SIZE

    transformed_features = pd.concat(transformed_features_batches, axis=0)
    return transformer, transformed_features




def benchmark_features_transform(
    X_train,
    y_train,
    X_test=None,
    group_train=None,
    group_test=None,
    feature_eng=None,
    feature_eng_parameters=None,
    debug=False,
):

    ### Numerai
    if feature_eng in [
        "numerai",
    ]:
        if feature_eng_parameters is None:
            feature_eng_parameters = {
                "usesquare": False,
                "no_product_features": 0,
                "seed": 10,
            }
        transformer = NumeraiTransformer(**feature_eng_parameters)


    if feature_eng is not None:
        extracted_features_train = transformer.transform(X_train, is_train=True)
        if X_test is not None:
            extracted_features_test = transformer.transform(X_test, is_train=False)
        else:
            extracted_features_test = None

        return transformer, extracted_features_train, extracted_features_test
    else:
        if X_test is not None:
            return None, X_train, X_test
        else:
            return None, X_train, None
