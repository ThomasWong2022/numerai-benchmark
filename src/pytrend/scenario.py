#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Scenario Analysis for finance
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
import os
from scipy.stats import norm

from .asset import Asset, YahooAsset, Compustat_CRSP_Data, Compustat_Data
from .util import strategy_metrics, align_features_target, GroupedTimeSeriesSplit
from .option import option_replicate, benchmark_options
from .finance import volatility_target, benchmark_volatility_target
from .portfolio import factor_construction, index_construction


###
### Technical Analysis
###
### Use option replication and volatility scaling model to reproduce standard trend-following and mean-reversion based portfolio
###
### Given percentage return of a stock and a hedge
### stock_return: pd.Series
### hedge_return: pd.Series
### create_hedges: Boolean (Whether to consider the spread between the stock and hedge as a new asset)
### long_only: Boolean (Whether to have long positions in stock, as some illuquid stocks cannot be shorted)
###
###


def technical_analysis_scenario(
    stock_return,
    hedge_return,
    stock="AAPL",
    hedge="SPY",
    tcost=0.001,
    leverage_cap=1,
    long_only=False,
    hedge_ratio=1,
):
    delta_list = list()
    strategy_return_list = list()
    strat_prefix = "{}_".format(stock)
    ## Option Strategies
    deltas, strat_return, strat_perform, strat_greeks = benchmark_options(
        stock_return,
        long_only=long_only,
        tcost=tcost,
        hedge_ratio=hedge_ratio,
    )
    deltas = deltas.add_prefix(strat_prefix)
    strat_return = strat_return.add_prefix(strat_prefix)
    strat_perform.index = [strat_prefix + x for x in strat_perform.index]
    delta_list.append(deltas)
    strategy_return_list.append(strat_return)

    ## Prepare Summary
    strategy_deltas = pd.concat(delta_list, axis=1)
    strategy_returns = pd.concat(strategy_return_list, axis=1)
    strategy_performance = pd.DataFrame(
        dict(strategy_returns.dropna().apply(strategy_metrics, axis=0))
    ).transpose()
    return strategy_deltas, strategy_returns, strategy_performance, strat_greeks


def optimal_technical_analysis(
    stock_return, hedge_return, stock, hedge, model_params, debug=False
):
    hedge_return = hedge_return.reindex(stock_return.index)
    deltas, strat, performance, greeks = technical_analysis_scenario(
        stock_return,
        hedge_return,
        stock,
        hedge,
        tcost=model_params["tcost"],
        hedge_ratio=model_params["hedge_ratio"],
        long_only=model_params["long_only"],
    )
    ## Cross Validation split
    tscv = GroupedTimeSeriesSplit(
        n_splits=model_params["n_splits"],
        valid_splits=model_params["valid_splits"],
        max_train_size=model_params["max_train_size"],
        gap=model_params["gap"],
    )
    ## Generate Portfolio Deltas from cross validation
    portfolio_deltas_good = list()
    portfolio_deltas_bad = list()

    for train_index, test_index in tscv.split(strat.dropna()):
        if debug:
            print(train_index.shape, test_index.shape)
            print(train_index[-1], test_index[0])
        strategy_performance = pd.DataFrame(
            dict(strat.dropna().loc[train_index].apply(strategy_metrics, axis=0))
        ).transpose()
        strategy_performance.index = strat.columns
        strategy_performance.drop(["{}_buyandhold".format(stock)], axis=0, inplace=True)
        best_strategy = (
            strategy_performance.sort_values("sharpe", ascending=False)
            .head(model_params["best_strategy"])
            .index
        )
        bad_strategy = (
            strategy_performance.sort_values("sharpe", ascending=True)
            .head(model_params["best_strategy"])
            .index
        )
        portfolio_deltas_good.append(
            deltas.fillna(0).loc[test_index, best_strategy].mean(axis=1)
        )
        portfolio_deltas_bad.append(
            deltas.fillna(0).loc[test_index, bad_strategy].mean(axis=1)
        )
        if debug:
            print(best_strategy, worst_strategy)
    best_portfolio_deltas = pd.concat(portfolio_deltas_good, axis=0).dropna()
    best_portfolio_deltas = best_portfolio_deltas.apply(lambda x: np.clip(x, -1, 1))
    worst_portfolio_deltas = pd.concat(portfolio_deltas_bad, axis=0).dropna()
    worst_portfolio_deltas = worst_portfolio_deltas.apply(lambda x: np.clip(x, -1, 1))
    ## Calculate performance
    transaction = best_portfolio_deltas.diff().abs() * model_params["tcost"]
    best_portfolio = (
        best_portfolio_deltas * stock_return.loc[best_portfolio_deltas.index].shift(-2)
    ).shift(2) - transaction.shift(1)
    best_portfolio.dropna(inplace=True)
    strategy_performance = strategy_metrics(best_portfolio)
    strategy_performance["correlation_stock"] = np.corrcoef(
        best_portfolio, stock_return.loc[best_portfolio.index].values
    )[0, 1]
    strategy_performance["correlation_hedge"] = np.corrcoef(
        best_portfolio, hedge_return.loc[best_portfolio.index].values
    )[0, 1]
    strategy_performance["stock"] = stock
    strategy_performance["hedge"] = hedge
    transaction = worst_portfolio_deltas.diff().abs() * model_params["tcost"]
    contrarian_portfolio = (
        worst_portfolio_deltas
        * stock_return.loc[worst_portfolio_deltas.index].shift(-2)
    ).shift(2) - transaction.shift(1)
    contrarian_portfolio.dropna(inplace=True)
    antistrategy_performance = strategy_metrics(contrarian_portfolio)
    antistrategy_performance["correlation_stock"] = np.corrcoef(
        contrarian_portfolio, stock_return.loc[contrarian_portfolio.index].values
    )[0, 1]
    antistrategy_performance["correlation_hedge"] = np.corrcoef(
        contrarian_portfolio, hedge_return.loc[contrarian_portfolio.index].values
    )[0, 1]
    antistrategy_performance["stock"] = stock
    antistrategy_performance["hedge"] = hedge
    return (
        best_portfolio_deltas,
        best_portfolio,
        strategy_performance,
        worst_portfolio_deltas,
        contrarian_portfolio,
        antistrategy_performance,
    )


### Analyse return of a stock using machine learning
###
### Run machine learning pipelines to benchmark prediction performance
### stock_target: pd.Series (Average forward return of the underlying asset)
### features: pd.DataFrame (Price and Volume data of different assets)
###
###
###
###
def machine_learning_scenario(
    features,
    target,
    tcost=0.001,
    leverage_cap=1,
    long_only=False,
):
    ### Process Price Data
    ### Check adjusted_close, adjusted_open, adjusted_high, adjusted_low and volume are in the dataframe
    ### Run sktime.Imputer to impute missing values
    ### Generate statistics based features (RollingSummaryTransformer)
    ###
    ###
    return None


###
### Factor Portfolios
###
### Create Factor Portfolios using data from CRSP and Compustat
###
###


### Create Sector to Industry Map and Industry Group to Industry Map


def __create_factor_maps(industry_code="../../data/Compustat_industry_code_2020.csv"):

    industrycode = pd.read_csv(industry_code)
    subindustrycodes = industrycode[industrycode["gictype"] == "GSUBIND"]
    subindustrycodes["Industry"] = subindustrycodes["giccd"].apply(lambda x: x // 100)
    subindustrycodes["IndustryGroup"] = subindustrycodes["giccd"].apply(
        lambda x: x // 10000
    )
    subindustrycodes["Sector"] = subindustrycodes["giccd"].apply(lambda x: x // 1000000)

    SectorMap = dict()
    for sec in subindustrycodes["Sector"].unique():
        SectorMap[sec] = list()
    for i, row in subindustrycodes.iterrows():
        SectorMap[row["Sector"]].append(row["giccd"])

    IndGroupMap = dict()
    for sec in subindustrycodes["IndustryGroup"].unique():
        IndGroupMap[sec] = list()
    for i, row in subindustrycodes.iterrows():
        IndGroupMap[row["IndustryGroup"]].append(row["giccd"])

    IndustryMap = dict()
    for sec in subindustrycodes["Industry"].unique():
        IndustryMap[sec] = list()
    for i, row in subindustrycodes.iterrows():
        IndustryMap[row["Industry"]].append(row["giccd"])

    return SectorMap, IndGroupMap, IndustryMap


def fama_french_factor(
    datadf,
    style,
    factor_name,
    quantile=5,
    tcost=0.005,
    equal_weighted=False,
    rebalance_freq=63,
    min_assets=0,
    max_assets=500,
):

    style_map = {
        "size": "market_cap",
        "volatility": "volatility",
        "momentum": "momentum",
        "value": "bp_ratio",
        "carry": "shareholder_yield",
        "quality": "earnings_yield",
        "crowding": "short_interest",
        "flow": "tradingflow",
        "beta": "beta",
    }

    ## For Value and Carry stocks filter universe to those with valid fundamentals data

    if not equal_weighted:
        (
            asset_quantile,
            quantile_count,
            quantile_weights,
            index_return,
            tcosts,
        ) = factor_construction(
            datadf["return"],
            datadf[style_map[style]],
            tcost=tcost,
            factor_name=factor_name,
            use_asset_weights=True,
            weights=datadf["market_cap"],
            quantile=quantile,
            trading_dates=datadf["trading_days"],
            rebalance_freq=rebalance_freq,
            min_assets=min_assets,
            max_assets=max_assets,
        )
    else:
        (
            asset_quantile,
            quantile_count,
            quantile_weights,
            index_return,
            tcosts,
        ) = factor_construction(
            datadf["return"],
            datadf[style_map[style]],
            tcost=tcost,
            factor_name=factor_name,
            use_asset_weights=False,
            quantile=quantile,
            trading_dates=datadf["trading_days"],
            rebalance_freq=rebalance_freq,
            min_assets=min_assets,
            max_assets=max_assets,
        )

    if index_return.shape[0] > 0:
        return asset_quantile, quantile_count, quantile_weights, index_return, tcosts
    else:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


def compustat_factor(
    subindustry,
    universe="sector",
    market="US",
    style="size",
    startyear=None,
    endyear=None,
    quantile=5,
    equal_weighted=False,
    pricedataonly=True,
    tcost=0.005,
    rebalance_freq=63,
    folder="../../data/Compustat",
    fundamentals_folder="../../data/Compustat_Fundamentals",
    usnameref="../../data/Compustat_metadata_2020.csv",
    industry_code="../../data/Compustat_industry_code_2020.csv",
    debug=False,
):

    default_settings = {
        "CHN": {
            "cap": 1e11,
            "start": 2006,
            "end": 2021,
        },
        "HKG": {
            "cap": 1e10,
            "start": 2002,
            "end": 2021,
        },
        "CHE": {
            "cap": 2e9,
            "start": 2002,
            "end": 2021,
        },
        "DEU": {
            "cap": 2e9,
            "start": 2002,
            "end": 2021,
        },
        "JPN": {
            "cap": 1e11,
            "start": 2005,
            "end": 2021,
        },
        "AUS": {
            "cap": 2e9,
            "start": 2002,
            "end": 2021,
        },
        "GBR": {
            "cap": 1e9,
            "start": 2002,
            "end": 2021,
        },
        "US": {
            "cap": 1e9,
            "start": 2002,
            "end": 2021,
        },
        "default": {
            "cap": 1e9,
            "start": 2002,
            "end": 2021,
        },
    }

    if not startyear:
        startyear = default_settings.get(market, "default")["start"]
    if not endyear:
        endyear = default_settings.get(market, "default")["end"]

    if equal_weighted:
        col_name = "Compustat_{}_{}_{}_{}_factor_eqweighted".format(
            market,
            style,
            universe,
            subindustry,
        )
    else:
        col_name = "Compustat_{}_{}_{}_{}_factor_capweighted".format(
            market,
            style,
            universe,
            subindustry,
        )

    SectorMap, IndGroupMap, IndustryMap = __create_factor_maps(
        industry_code=industry_code
    )

    if universe == "sector":
        selected_industry = SectorMap[subindustry]
    elif universe == "industry_group":
        selected_industry = IndGroupMap[subindustry]
    elif universe == "industry":
        selected_industry = IndustryMap[subindustry]
    elif universe == "subindustry":
        selected_industry = [subindustry]
    elif universe == "all":
        selected_industry = None

    datadf = Compustat_Data(
        market=market,
        startyear=startyear,
        endyear=endyear,
        subinds=selected_industry,
        folder=folder,
        fundamentals_folder=fundamentals_folder,
        usnameref=usnameref,
        quantile=quantile,
        debug=debug,
    )

    if datadf.shape[0] > 0:
        (
            asset_quantile,
            quantile_count,
            quantile_weights,
            index_return,
            tcosts,
        ) = fama_french_factor(
            datadf,
            style,
            col_name,
            quantile=quantile,
            tcost=tcost,
            equal_weighted=equal_weighted,
            rebalance_freq=rebalance_freq,
        )
        return asset_quantile, quantile_count, quantile_weights, index_return, tcosts
    else:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


def crsp_factor(
    subindustry,
    universe="sector",
    market="USCore",
    style="size",
    startyear=1921,
    endyear=2021,
    quantile=5,
    equal_weighted=False,
    min_assets=0,
    max_assets=500,
    folder="../../data/CRSP",
    fundamentals_folder="../../data/Compustat_Fundamentals",
    linktable="../../data/Compustat_CRSP_link_2020.csv",
    crsptable="../../data/CRSP_names_2020.csv",
    industry_code="../../data/Compustat_industry_code_2020.csv",
    tcost=0.005,
    rebalance_freq=63,
    use_fundamentals=False,
    use_option_volume=False,
    use_vol_surface=False,
    debug=False,
):

    if equal_weighted:
        col_name = "CRSP_{}_{}_{}_{}_factor_eqweighted".format(
            market,
            style,
            universe,
            subindustry,
        )
    else:
        col_name = "CRSP_{}_{}_{}_{}_factor_capweighted".format(
            market,
            style,
            universe,
            subindustry,
        )

    SectorMap, IndGroupMap, IndustryMap = __create_factor_maps(
        industry_code=industry_code
    )

    if universe == "sector":
        selected_industry = SectorMap[subindustry]
    elif universe == "industry_group":
        selected_industry = IndGroupMap[subindustry]
    elif universe == "industry":
        selected_industry = IndustryMap[subindustry]
    elif universe == "subindustry":
        selected_industry = [subindustry]
    elif universe == "all":
        selected_industry = None

    datadf = Compustat_CRSP_Data(
        folder=folder,
        fundamentals_folder=fundamentals_folder,
        linktable=linktable,
        crsptable=crsptable,
        market=market,
        subinds=selected_industry,
        startyear=startyear,
        endyear=endyear,
        use_fundamentals=use_fundamentals,
        use_option_volume=use_option_volume,
        use_vol_surface=use_vol_surface,
        quantile=quantile,
        debug=debug,
    )

    if datadf.shape[0] > 0:
        (
            asset_quantile,
            quantile_count,
            quantile_weights,
            index_return,
            tcosts,
        ) = fama_french_factor(
            datadf,
            style,
            col_name,
            quantile=quantile,
            min_assets=min_assets,
            max_assets=max_assets,
            tcost=tcost,
            equal_weighted=equal_weighted,
            rebalance_freq=rebalance_freq,
        )
        return asset_quantile, quantile_count, quantile_weights, index_return, tcosts
    else:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


def crsp_index(
    subindustry="all",
    universe="all",
    market="USCore",
    startyear=1921,
    endyear=2021,
    folder="../../data/CRSP",
    fundamentals_folder="../../data/Compustat_Fundamentals",
    linktable="../../data/Compustat_CRSP_link_2020.csv",
    crsptable="../../data/CRSP_names_2020.csv",
    industry_code="../../data/Compustat_industry_code_2020.csv",
    tcost=0.005,
    rebalance_freq=63,
    min_assets=0,
    max_assets=500,
    returns_col="return",
    weights_col="market_cap",
    use_fundamentals=False,
    use_option_volume=False,
    use_vol_surface=False,
    debug=False,
):

    SectorMap, IndGroupMap, IndustryMap = __create_factor_maps(
        industry_code=industry_code
    )

    if universe == "sector":
        selected_industry = SectorMap[subindustry]
    elif universe == "industry_group":
        selected_industry = IndGroupMap[subindustry]
    elif universe == "industry":
        selected_industry = IndustryMap[subindustry]
    elif universe == "subindustry":
        selected_industry = [subindustry]
    elif universe == "all":
        selected_industry = None

    if min_assets > 0:
        col_name = "CRSP_{}_{}_{}_{}_{}_index".format(
            market,
            universe,
            subindustry,
            min_assets,
            max_assets,
        )
    else:
        col_name = "CRSP_{}_{}_{}_{}_index".format(
            market,
            universe,
            subindustry,
            max_assets,
        )

    datadf = Compustat_CRSP_Data(
        folder=folder,
        fundamentals_folder=fundamentals_folder,
        linktable=linktable,
        crsptable=crsptable,
        market=market,
        subinds=selected_industry,
        startyear=startyear,
        endyear=endyear,
        use_fundamentals=use_fundamentals,
        use_option_volume=use_option_volume,
        use_vol_surface=use_vol_surface,
        debug=debug,
    )

    if datadf.shape[0] > 0:
        (asset_weights, index_return, tcosts,) = index_construction(
            asset_return=datadf[returns_col],
            asset_feature=datadf[weights_col],
            trading_dates=datadf["trading_days"],
            min_assets=min_assets,
            max_assets=max_assets,
            index_name=col_name,
            tcost=tcost,
            rebalance_freq=rebalance_freq,
        )
        return asset_weights, index_return, tcosts
    else:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
