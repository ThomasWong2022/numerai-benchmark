"""
Numerai Signals 
"""


from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

import pycatch22

from .feature import SignatureTransformer


## Feature Transformation per era
def normalise_features_era(df, feature_cols, group_labels=None, keep_original=False):
    transformed_features = list()
    if group_labels is not None:
        for group in group_labels:
            group_features = list()
            for i, df_group in df.groupby(group):
                df_group_ranked = df_group[feature_cols].rank(pct=True, axis=0) - 0.5
                df_group_ranked.fillna(0, inplace=True)
                df_group_ranked = df_group_ranked * 5
                df_group_ranked.columns = [
                    "{}_{}_ranked".format(x, group) for x in feature_cols
                ]
                group_features.append(pd.concat([df_group_ranked], axis=1))
            group_features_df = pd.concat(group_features, axis=0)
            transformed_features.append(group_features_df)
    ## On All Data
    df_ranked = df[feature_cols].rank(pct=True, axis=0) - 0.5
    df_ranked.fillna(0, inplace=True)
    df_ranked = df_ranked * 5
    df_ranked.columns = ["{}_ranked".format(x) for x in feature_cols]
    transformed_features.append(df_ranked)
    if keep_original:
        transformed_features.append(df[feature_cols])
    transformed_df = pd.concat(transformed_features, axis=1)
    return transformed_df


class NumeraiSignatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, signature_level=4, lookbacks=[4, 12, 52]):
        self.signature_level = signature_level
        self.lookbacks = lookbacks
        self.signaturetransformers = dict()
        for lookback in lookbacks:
            self.signaturetransformers[lookback] = SignatureTransformer(
                lookback, signature_level
            )

    def transform(self, X):

        signature_outputs = list()
        for lookback in self.lookbacks:
            signature_outputs.append(
                self.signaturetransformers[lookback].transform(X.dropna())
            )

        output = pd.concat(signature_outputs, axis=1)

        return output.dropna().add_prefix("feature_")


class NumeraiStatsTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        lookbacks=[
            4,
            12,
            52,
        ],
    ):

        self.lookbacks = lookbacks

    def transform(self, X):

        stats_outputs = list()
        for lookback in self.lookbacks:
            for operation in ["mean", "std", "skew", "kurt"]:
                agg_X = getattr(X.rolling(lookback), operation)()
                agg_X.columns = [f"{col}_{operation}_{lookback}" for col in X.columns]
                stats_outputs.append(agg_X)
        output = pd.concat(stats_outputs, axis=1)

        return output.dropna().add_prefix("feature_")


def process_single_stock(
    df,
    shift=4,
    live_data=False,
    selected_cols=[
        "target_20d",
        "target_20d_raw_return",
        "target_20d_factor_neutral",
        "target_20d_factor_feat_neutral",
    ],
):
    shift_df = df[selected_cols].shift(shift) - 0.5
    if not live_data:
        ## Add Basic Stats Features
        transformer = NumeraiStatsTransformer()
        stats_features = transformer.transform(shift_df)
        features = stats_features
        features[selected_cols] = df[selected_cols]
        return features.dropna()
    else:
        MAX_LOOKBACK = 252
        transformer = NumeraiStatsTransformer()
        stats_features = transformer.transform(shift_df.tail(MAX_LOOKBACK))
        features = stats_features
        return features.tail(1)
