#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Replicating Option Strategies
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
from scipy.stats import norm

from .util import align_features_target, strategy_metrics

### Option Replication
###
###
### Deriving Black-Scholes prices and greeks for an European option given strike and volatility
### Interest rates is assumed to be zero


### stock_return: pd.Series (timestamp x ) historical daily percentage return of an asset
### moneyness: pd.Series (timestamp x )
### maturity: pd.Series/float
### implied_vol: pd.Series
###


def black_scholes_pricer(
    stock_return,
    moneyness,
    implied_vol,
    maturity,
    vol_multiplier=1,
    option_type="call",
):
    ### Calcuate Strike
    df = pd.DataFrame(stock_return).copy()
    df.columns = ["return"]
    df["moneyness"] = np.clip(moneyness, 0.05, 20)
    df["adjusted_price"] = (1 + df["return"]).cumprod()
    df["strike"] = df["adjusted_price"] * df["moneyness"]

    ## Volatility Smile
    df["vol"] = implied_vol
    df["vol"] = df["vol"] * np.power((1 + np.abs(df["moneyness"] - 1)), vol_multiplier)

    ## Maturity of option
    df["maturity"] = maturity

    ## Calculate Option delta
    df["d1"] = (
        1
        / (df["vol"] * np.sqrt(df["maturity"]))
        * (
            np.log(df["adjusted_price"])
            - np.log(df["strike"])
            + df["vol"] * df["vol"] / 2 * df["maturity"]
        )
    )
    df["d2"] = df["d1"] - df["vol"] * np.sqrt(df["maturity"])
    df["delta_call"] = norm.cdf(df["d1"])
    if option_type == "call":
        df["delta"] = df["delta_call"]
    else:
        ## Put Call Parity
        df["delta"] = df["delta_call"] - 1
    ## Other Greeks
    df["gamma"] = norm.pdf(df["d1"]) / (
        df["adjusted_price"] * df["vol"] * np.sqrt(df["maturity"])
    )
    df["vega"] = df["adjusted_price"] * norm.pdf(df["d1"]) * np.sqrt(df["maturity"])
    df["theta"] = (
        -1
        * norm.pdf(df["d1"])
        * df["vol"]
        * df["adjusted_price"]
        / (2 * np.sqrt(df["maturity"]))
    )
    ## Theorectical price
    if option_type == "call":
        df["bsprice"] = (
            norm.cdf(df["d1"]) * df["adjusted_price"]
            - norm.cdf(df["d2"]) * df["strike"]
        )
        df["intrinsic_value"] = np.maximum(df["adjusted_price"] - df["strike"], 0)
    else:
        df["bsprice"] = (
            norm.cdf(df["d1"]) * df["adjusted_price"]
            - norm.cdf(df["d2"]) * df["strike"]
            + df["strike"]
            - df["adjusted_price"]
        )
        df["intrinsic_value"] = np.maximum(df["strike"] - df["adjusted_price"], 0)
    df["time_value"] = df["bsprice"] - df["intrinsic_value"]

    return df[
        [
            "return",
            "delta",
            "gamma",
            "vega",
            "theta",
            "intrinsic_value",
            "bsprice",
        ]
    ]


### Option Replication
###
### We assume as the trade signal is calculated at the end of day T, trade is entered at the end of day T+1
###
### stock: pd.Series (timestamp x ) historical daily percentage return of an asset
### option_type: str (call or put)
### strike_type: str (fixed_quarter, fixed_year, lookback)
### moneyness: float (multiplier to calculate reference strike from historical prices)
### maturity: float (number of years until option matures)
### historical_vol: float (number of years to lookback to calculate historical volatility)
### vol_multiple: float (volatility smile adjustment)
### hedge_ratio: float (neutralise market exposure)
### lookback: int (number of days to lookback)
### tcost: float (transaction costs)
###
###
### Returns: dataframe of option greeks and prices, linear combinations of columns can be used to represent different strategies


def option_replicate(
    stock,
    option_type="call",
    std=0,
    maturity=1,
    historical_vol=1,
    vol_multiplier=1,
    hedge_ratio=0,
    lookback=252,
    tcost=0.001,
):
    ### Create price series
    df = pd.DataFrame(stock).copy()
    df.columns = ["return"]
    df["adjusted_price"] = (1 + df["return"]).cumprod()
    df["standard_deviation"] = (
        df["adjusted_price"].rolling(int(historical_vol * 252)).std()
    )

    ## Calculate historical volatilty
    df["log_return"] = np.log(1 + df["return"])
    df["vol"] = df["log_return"].rolling(int(historical_vol * 252)).std() * np.sqrt(252)
    ## Calculate strike and maturity
    df["strike"] = (
        df["adjusted_price"].rolling(lookback, min_periods=1).mean()
        + df["standard_deviation"] * std
    )
    df["maturity"] = maturity

    ## Volatility smile adjustement
    df["moneyness"] = df["strike"] / df["adjusted_price"]

    df = black_scholes_pricer(
        df["return"],
        df["moneyness"],
        df["vol"],
        df["maturity"],
        vol_multiplier=vol_multiplier,
        option_type=option_type,
    )

    ## Rebalance
    df["hedged_delta"] = (
        df["delta"] - hedge_ratio * df["delta"].rolling(240, min_periods=1).mean()
    )
    df["hedged_delta"] = df["hedged_delta"].rolling(5).mean()
    df["tcost"] = df["hedged_delta"].diff().abs() * tcost
    ## -2 for buying at next day close
    df["option_forward_return"] = df["hedged_delta"] * df["return"].shift(-2)
    df["option_return"] = df["option_forward_return"].shift(2) - df["tcost"].shift(1)

    return df


### Option Strategies
##
## Example option strategies that are commonly used
###


def trend_following(
    stock,
    lookback=252,
    maturity=0.25,
    historical_vol=1,
    hedge_ratio=0,
    tcost=0.001,
):
    option_1 = option_replicate(
        stock,
        option_type="call",
        std=0,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    option_2 = option_replicate(
        stock,
        option_type="put",
        std=0,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )

    strategy = option_1 + option_2
    return strategy


def collar_trade(
    stock,
    std=1,
    maturity=0.25,
    historical_vol=1,
    lookback=252,
    hedge_ratio=0,
    tcost=0.001,
):
    ## Synthetic Long stock by long call and short put at the same strike
    option_0 = option_replicate(
        stock,
        option_type="call",
        std=0,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    option_1 = option_replicate(
        stock,
        option_type="put",
        std=0,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    option_2 = option_replicate(
        stock,
        option_type="put",
        std=-1 * std,
        vol_multiplier=1.2,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    option_3 = option_replicate(
        stock,
        option_type="call",
        std=std,
        vol_multiplier=1.2,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    strategy = option_0 - option_1 + option_2 - option_3
    return strategy


def mean_reversion(
    stock,
    std=2,
    maturity=1,
    historical_vol=1,
    lookback=63,
    hedge_ratio=0,
    tcost=0.001,
):
    ## Short Put
    option_1 = option_replicate(
        stock,
        option_type="put",
        std=-1 * std,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    ## Buy Further Out of Money Put to Cap Loss
    option_2 = option_replicate(
        stock,
        option_type="put",
        std=-2 * std,
        vol_multiplier=1,
        hedge_ratio=hedge_ratio,
        maturity=maturity,
        historical_vol=historical_vol,
        lookback=lookback,
        tcost=tcost,
    )
    strategy = option_2 - option_1
    return strategy


### Benchmark Options
###
### Grid Search for various option strategies
###
###
### stock: pd.Series (timestamp x ) historical daily percentage return of an asset
###


def __calculate_option_benchmark(
    strat, strategy_delta, strategy_return, long_only, strategy_name, tcost
):
    if not long_only:
        strategy_delta[strategy_name] = np.clip(strat["hedged_delta"], -1, 1)
    else:
        strategy_delta[strategy_name] = np.clip(strat["hedged_delta"], 0, 1)
    transaction = strategy_delta[strategy_name].diff().abs() * tcost
    strategy_return[strategy_name] = (
        strategy_delta[strategy_name] * strategy_return["buyandhold"].shift(-2)
    ).shift(2).values - transaction.shift(1).values
    return strategy_delta, strategy_return


def benchmark_options(stock, tcost=0.001, long_only=False, hedge_ratio=0):

    strategy_delta = pd.DataFrame(index=stock.index)
    strategy_return = pd.DataFrame(index=stock.index)
    strategy_greeks = pd.DataFrame(index=stock.index)

    ## buy_and_hold
    strategy_delta["buyandhold"] = 1
    strategy_return["buyandhold"] = stock.values

    greek_cols = [
        "delta",
        "gamma",
        "theta",
        "vega",
    ]

    ## Collar
    if False:
        for lookback in [240 * i for i in range(1, 2, 1)]:
            for moneyness in [1 * i for i in range(1, 4, 1)]:
                strategy_name = "collar_{}_{}".format(lookback, np.round(moneyness, 3))
                strat = collar_trade(
                    stock,
                    std=moneyness,
                    tcost=tcost,
                    lookback=lookback,
                    hedge_ratio=hedge_ratio,
                )
                strategy_delta, strategy_return = __calculate_option_benchmark(
                    strat,
                    strategy_delta,
                    strategy_return,
                    long_only,
                    strategy_name,
                    tcost,
                )
                for greek in greek_cols:
                    strategy_greeks["{}_{}".format(strategy_name, greek)] = strat[greek]

    ## Trend Following
    for lookback in [10 * i for i in range(15, 71, 1)]:
        for historical_vol in [1 * i for i in range(1, 2, 1)]:
            for maturity in [1 * i for i in range(3, 4, 1)]:
                strategy_name = "trend_{}_{}_{}".format(
                    lookback, historical_vol, maturity
                )
                strat = trend_following(
                    stock,
                    lookback=lookback,
                    historical_vol=historical_vol,
                    maturity=maturity,
                    tcost=tcost,
                    hedge_ratio=hedge_ratio,
                )
                strategy_delta, strategy_return = __calculate_option_benchmark(
                    strat,
                    strategy_delta,
                    strategy_return,
                    long_only,
                    strategy_name,
                    tcost,
                )
                for greek in greek_cols:
                    strategy_greeks["{}_{}".format(strategy_name, greek)] = strat[greek]

    ## Mean Reversion (Short Put Spread)
    if False:
        for lookback in [240 * i for i in range(1, 2, 1)]:
            for moneyness in [1 * i for i in range(1, 4, 1)]:
                strategy_name = "meanreversion_{}_{}".format(
                    lookback, np.round(moneyness, 3)
                )
                strat = mean_reversion(
                    stock,
                    std=moneyness,
                    tcost=tcost,
                    lookback=lookback,
                    hedge_ratio=hedge_ratio,
                )
                strategy_delta, strategy_return = __calculate_option_benchmark(
                    strat,
                    strategy_delta,
                    strategy_return,
                    long_only,
                    strategy_name,
                    tcost,
                )

                for greek in greek_cols:
                    strategy_greeks["{}_{}".format(strategy_name, greek)] = strat[greek]

    ## Strategy Metrics
    strategy_performance = pd.DataFrame(
        dict(strategy_return.dropna().apply(strategy_metrics, axis=0))
    ).transpose()
    strategy_performance.index = strategy_return.columns
    return strategy_delta, strategy_return, strategy_performance, strategy_greeks
