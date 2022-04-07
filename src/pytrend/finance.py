#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# A collection of statistical models from finance
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
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .util import align_features_target, strategy_metrics
from .option import option_replicate

### Volatility target
##
## Given daily return of a strategy, calculate the vol-target position (capped at 3x leverage) and vol-target returns
##
##
##
##
def volatility_target(
    stock,
    interval=1,
    vol_lookback=63,
    vol_target=0.2,
    vol_power=2,
    leverage_cap=1,
    tcost=0.001,
):
    df = pd.DataFrame(stock).copy()
    df.columns = ["return"]
    df["forward_return"] = df["return"].shift(-2)
    volatility = stock.rolling(vol_lookback).std() * np.sqrt(252 / interval)
    df["delta"] = np.clip(
        np.power(vol_target / volatility, vol_power), -1 * leverage_cap, leverage_cap
    )
    df["tcost"] = df["delta"].diff().abs() * tcost
    df["voladj_forward_return"] = df["delta"] * df["forward_return"]
    df["vol_target_return"] = df["voladj_forward_return"].shift(2) - df["tcost"].shift(
        1
    )
    return df[["delta", "vol_target_return"]]


def benchmark_volatility_target(
    stock,
    tcost=0.001,
    leverage_cap=1,
):
    strategy_delta = pd.DataFrame(index=stock.index)
    strategy_return = pd.DataFrame(index=stock.index)
    ## buy_and_hold
    strategy_delta["buyandhold"] = 1
    strategy_return["buyandhold"] = stock.values
    ## Different volatility
    for vol in [0.1 * i for i in range(1, 4)]:
        strategy_name = "voltarget_{}".format(np.round(vol, 3))
        strat = volatility_target(
            stock, vol_target=vol, leverage_cap=leverage_cap, tcost=tcost
        )
        strategy_delta[strategy_name] = strat["delta"]
        strategy_return[strategy_name] = strat["vol_target_return"]
        ## Strategy Metrics
    strategy_performance = pd.DataFrame(
        dict(strategy_return.dropna().apply(strategy_metrics, axis=0))
    ).transpose()
    return strategy_delta, strategy_return, strategy_performance


### Butterfly Strategy
###
### Given forward prediction of asset return, create a trading strategy based on call butterfly with peak around predicted return
####
#### stock: pd.Series (percentage return of stock)
#### prediction: pd.Series (prediced percentage return of stock)
####
###
###
def butterfly_prediction(
    stock,
    prediction,
    maturity=5,
    historical_vol=1,
    vol_multiplier=1,
    tcost=0.001,
    lookback=5,
):

    ## Calculate standard deviation from prediction
    df = pd.DataFrame(stock).copy()
    df.columns = ["return"]
    df["log_return"] = np.log(1 + df["return"])
    df["vol"] = df["log_return"].rolling(int(historical_vol * 252)).std() * np.sqrt(252)
    df["std"] = prediction / df["vol"]

    option_1 = option_replicate(
        stock,
        std=0,
        maturity=maturity,
        historical_vol=historical_vol,
        vol_multiplier=vol_multiplier,
        lookback=lookback,
        tcost=tcost,
    )
    option_2 = option_replicate(
        stock,
        std=df["std"],
        maturity=maturity,
        historical_vol=historical_vol,
        vol_multiplier=vol_multiplier,
        lookback=lookback,
        tcost=tcost,
    )
    option_3 = option_replicate(
        stock,
        std=df["std"] * 2,
        maturity=maturity,
        historical_vol=historical_vol,
        vol_multiplier=vol_multiplier,
        lookback=lookback,
        tcost=tcost,
    )
    strat = option_1 - 2 * option_2 + option_3
    return strat


#### Transaction costs analysis
###
### A very simple model for transaction costs with a constant percentage of fees and using moving average to smooth predictions
###
### stock: pd.Series (historical percertange return)
### delta: pd.Series (historical positions)
### lookback: int (number of days to compute moving average of delta)
### tcost: float (percentage fees for transactions)
###
def transaction_cost_model(stock, delta, lookback=10, tcost=0.001):
    df = pd.DataFrame(stock).copy()
    df.columns = ["return"]
    df["forward_return"] = df["return"].shift(-2)
    df["delta"] = delta.rolling(lookback, min_periods=1).mean()
    df["tcost"] = df["delta"].diff().abs() * tcost
    df["smoothed_forward_return"] = df["delta"] * df["forward_return"]
    df["smoothed_return"] = df["smoothed_forward_return"].shift(2) - df["tcost"].shift(
        1
    )
    return df[["delta", "smoothed_return"]]


def benchmark_transaction_cost_model(stock, delta, tcost=0.001):
    strategy_delta = pd.DataFrame(index=stock.index)
    strategy_return = pd.DataFrame(index=stock.index)
    ## buy_and_hold
    strategy_delta["buy_and_hold"] = 1
    strategy_return["buy_and_hold"] = stock.values
    ## Different volatility
    for lookback in [5 * i for i in range(0, 11)]:
        if lookback == 0:
            lookback += 1
        strategy_name = "MA_{}".format(lookback)
        strat = transaction_cost_model(
            stock,
            delta,
            tcost=tcost,
            lookback=lookback,
        )
        strategy_delta[strategy_name] = strat["delta"]
        strategy_return[strategy_name] = strat["smoothed_return"]
        ## Strategy Metrics
    strategy_performance = pd.DataFrame(
        dict(strategy_return.dropna().apply(strategy_metrics, axis=0))
    ).transpose()
    return strategy_delta, strategy_return, strategy_performance


### Intraday data adjustment
####
## Given an unadjsuted intraday price data from US Market (Ex. CBOE short transactions) and an daily adjusted price from another data source (Ex. Yahoo Finance)
## Return the adjusted intraday time-series

##
## intraday_price pd.Series (raw price of an asset with pd.datetime index)
## adjusted_close pd.Series (adjusted close of an asset with pd.datetime index)


def adjust_intraday_ts(intraday_price, adjusted_close, market="US"):
    intra = pd.DataFrame(intraday_price).copy()
    intra.columns = ["raw_price"]
    daily = pd.DataFrame(adjusted_close).copy()
    daily.columns = ["adjusted_price"]
    intra["datetime"] = pd.to_datetime(intra.index)
    intra["date"] = intra["datetime"].dt.date
    intra["time"] = intra["datetime"].dt.time
    daily["datetime"] = pd.to_datetime(daily.index)
    daily["date"] = daily["datetime"].dt.date
    ## Regular hours for US Market
    if market == "US":
        intra = intra[
            (intra["time"] <= pd.to_datetime("16:00:00").time())
            & (intra["time"] >= pd.to_datetime("09:30:00").time())
        ]
    intra_close = intra.groupby("date").tail(n=1)
    ratio = intra_close.merge(
        daily,
        how="left",
        left_on="date",
        right_on="date",
    )
    ratio["ratio"] = ratio["adjusted_price"] / ratio["raw_price"]
    intra = intra.merge(
        ratio[["date", "ratio"]], how="left", left_on="date", right_on="date"
    )
    intra["adjusted_price"] = intra["ratio"] * intra["raw_price"]
    return intra.set_index("datetime")[["adjusted_price", "raw_price"]].copy()
