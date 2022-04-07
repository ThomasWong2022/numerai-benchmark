#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Standard Methods to create portfolios
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
from scipy.optimize import minimize


### Basket construction
###
### Index construction (Long only)
### create portfolio with weights according to the features, keeping only those with max_assets (500) largest non-zero weights
###
### asset_return: pd.DataFrame(timestamp x n_assets) historical return
### asset_feature: pd.DataFrame(timestamp x n_assets) features to contruct weights Ex. market_cap, assumed to be non-negative
###
### TODO: Fix rebalance dates relative to the start of trading year instead of depending on the start of the trading date?
###
###
def index_construction(
    asset_return,
    asset_feature,
    rebalance_freq=63,
    trading_dates=None,
    min_assets=0,
    max_assets=500,
    index_name="US_500",
    tcost=0.005,
):
    ### Forward reutrns
    asset_returns = asset_return.shift(-2)
    ### Asset Weights
    asset_index = asset_returns.index
    asset_columns = asset_returns.columns
    ### Algin Return and Feature columns
    asset_feature = asset_feature.loc[:, asset_columns]
    ## Select the rebalance dates
    if type(trading_dates) != pd.Series:
        asset_feature_filtered = asset_feature.iloc[::rebalance_freq, :]
    else:
        if rebalance_freq > 42:
            ## Quarterly rebalance at the end of Feb, May, August and November as the tradition for most index
            rebalance_dates = trading_dates[
                (trading_dates % rebalance_freq == 40) & (trading_dates < 240)
            ].index
        else:
            rebalance_dates = trading_dates[
                (trading_dates % rebalance_freq == 0) & (trading_dates < 240)
            ].index
        asset_feature_filtered = asset_feature.loc[rebalance_dates, :]
    asset_weights = asset_feature_filtered.reindex(asset_index, method="pad")
    ### Rank Asset_weights
    ### Keep only max_assets with largest feature
    asset_weights_rank = asset_weights.rank(
        axis=1,
        ascending=False,
        method="min",
    )
    asset_weights_filter = asset_weights_rank.applymap(
        lambda x: np.where(x <= max_assets and x >= min_assets, 1, 0)
    )
    asset_weights = asset_weights * asset_weights_filter
    asset_weights_sum = asset_weights.sum(axis=1).apply(lambda x: x if x > 0 else 1e100)
    asset_weights = asset_weights.divide(asset_weights_sum, axis="index")
    index_returns = (asset_returns * asset_weights).sum(axis=1)
    transaction_costs = pd.Series(asset_weights.diff().abs().sum(axis=1) * tcost)
    transaction_costs.columns = ["transaction_cost"]
    index_returns = index_returns.shift(2) - transaction_costs.shift(1).values
    index_returns.columns = [index_name]

    return asset_weights, index_returns, transaction_costs.shift(1)


### Basket construction (Long/Short)
### sort assets into quantile(10) groups by feature, within each quantile crate equal weighted portfolio of assets
###
### asset_return: pd.DataFrame(timestamp x n_assets) historical return
### asset_feature: pd.DataFrame(timestamp x n_assets) features to contruct weights Ex. market_cap, assumed to be non-negative
### weights: pd.DataFrame(timestamp x n_assets) asset weights to use within each quantile
### trading_dates: pd.Series the trading day number within a year of
###


def factor_construction(
    asset_return,
    asset_feature,
    use_asset_weights=False,
    weights=None,
    min_assets=0,
    max_assets=500,
    rebalance_freq=63,
    trading_dates=None,
    quantile=5,
    factor_name="samplefactor",
    tcost=0.005,
):
    ### Forward reutrns
    asset_returns = asset_return.shift(-2)
    ### Asset Weights
    asset_index = asset_returns.index
    asset_columns = asset_returns.columns
    ### Algin Return and Feature columns
    asset_feature = asset_feature.loc[:, asset_columns]
    if use_asset_weights:
        weights = weights.loc[:, asset_columns]

    ## Select the rebalance dates
    if type(trading_dates) != pd.Series:
        asset_feature_filtered = asset_feature.iloc[::rebalance_freq, :]
    else:
        if rebalance_freq > 42:
            ## Quarterly rebalance at the end of Feb, May, August and November as the tradition for most index
            rebalance_dates = trading_dates[
                (trading_dates % rebalance_freq == 40) & (trading_dates < 240)
            ].index
        else:
            rebalance_dates = trading_dates[
                (trading_dates % rebalance_freq == 0) & (trading_dates < 240)
            ].index
        asset_feature_filtered = asset_feature.loc[rebalance_dates, :].copy()
    asset_feature_weights = asset_feature_filtered.reindex(asset_index, method="pad")

    ## Calculate asset_weights if needed
    if use_asset_weights:
        ## Normalise asset weights
        if type(trading_dates) != pd.Series:
            weights_filtered = weights.iloc[::rebalance_freq, :]
        else:
            weights_filtered = weights.loc[rebalance_dates, :]
        asset_weights = weights_filtered.reindex(asset_index, method="pad")
        ## Rank market_cap
        ### Keep only max_assets with largest feature
        asset_weights_rank = asset_weights.rank(
            axis=1,
            ascending=False,
            method="min",
        )
        asset_weights_filter = asset_weights_rank.applymap(
            lambda x: np.where(x <= max_assets and x >= min_assets, 1, np.nan)
        )

        asset_feature_weights = asset_feature_weights * asset_weights_filter

    ### Rank Asset_weights
    asset_quantile_raw = asset_feature_weights.rank(
        axis=1,
        ascending=True,
        method="max",
        pct=True,
    )
    asset_quantile = asset_quantile_raw.applymap(lambda x: x * quantile)
    asset_quantile = asset_quantile.applymap(np.ceil)

    ## Factor return
    factor_return = pd.DataFrame(index=asset_index)
    transaction_return = pd.DataFrame(index=asset_index)
    asset_quantile_all = list()
    for i in range(1, quantile + 1):
        factor_col = factor_name + "_{}".format(i)
        ## Create Asset Quantile Mask
        asset_quantile_mask = asset_quantile.applymap(lambda x: np.where(x == i, 1, 0))
        if use_asset_weights:
            asset_quantile_mask = asset_quantile_mask * weights
        ## Normalise Asset Quantile Mask
        asset_weights_rawsum = asset_quantile_mask.sum(axis=1).apply(
            lambda x: x if x > 0 else 1e100
        )
        asset_quantile_mask = asset_quantile_mask.div(
            asset_weights_rawsum, axis="index"
        )
        ## Transaction costs
        transaction_costs = asset_quantile_mask.diff().abs().sum(axis=1) * tcost
        price_return = (asset_returns * asset_quantile_mask).sum(axis=1)
        transaction_return[factor_col + "_tcost"] = transaction_costs.shift(1)
        factor_return[factor_col] = (
            price_return.shift(2) - transaction_return[factor_col + "_tcost"]
        )
        asset_quantile_all.append(asset_quantile_mask)
    ## Quantile Mask
    def clip(value, threshold):
        if value > threshold:
            return np.nan
        else:
            return value

    clipped_quantile = asset_quantile.applymap(lambda x: clip(x, threshold=quantile))

    quantile_count = pd.DataFrame(
        clipped_quantile.transpose().apply(lambda x: x.value_counts()).T.stack()
    )
    quantile_count.index.names = ["date", "quantile"]
    quantile_count.columns = ["count"]
    quantile_table = quantile_count.reset_index().pivot(
        index="date", columns="quantile", values="count"
    )

    valid_index = quantile_table.index

    quantile_weights = asset_quantile_all[0]
    for i in range(1, len(asset_quantile_all)):
        quantile_weights += asset_quantile_all[i]

    return (
        clipped_quantile.reindex(valid_index),
        quantile_table,
        quantile_weights.reindex(valid_index),
        factor_return.reindex(valid_index),
        transaction_return.reindex(valid_index),
    )
