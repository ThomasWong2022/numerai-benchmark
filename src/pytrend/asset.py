from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Type, Dict, List, Iterable

import pandas as pd
import numpy as np
import os.path
import yfinance as yf

import multiprocessing
from joblib import Parallel, delayed

import requests

from .compustat import (
    merge_compustat_dataset,
    read_crsp_price,
    read_compustat_us_fundamentals,
    read_compustat_us_short_interests,
    read_optionmetrics_us_volatility_surface,
)
from .compustat import (
    read_compustat_price,
    read_compustat_international_fundamentals,
)


### TODO: Use Python Data Class?

### Load price data, columns assumed to be multiindex with adjusted OHLCV with ticker names, index is a timestamp

### Asset for daily price data in the format of dividend adjusted OHLCV
class Asset:
    def __init__(self, pricedf: Optional[pd.DataFrame]):

        assert [
            "adjusted_close",
            "adjusted_high",
            "adjusted_low",
            "adjusted_open",
            "volume",
        ] in pricedf.columns.get_level_values(0)
        self.price = pricedf
        ### Compute return
        returns = self.price[["adjusted_close"]].pct_change()
        returns.rename(columns={"adjusted_close": "return"}, inplace=True)
        self.price = pd.concat([self.price, returns], axis=1)

        ## Compute derived metrics
        self._compute_features()

        return None

    def _compute_features(self):

        ## Ensure data column
        if not "date" in self.price.columns:
            self.price["date"] = self.price.index
        ## Seasonality Features
        self.price["dayofweek"] = self.price["date"].dt.dayofweek + 1
        self.price["month"] = self.price["date"].dt.month

    def compute_targets(self, forward_period=1):
        ## log_return
        log_return = np.log(1 + self.price["return"])
        forward_return = (
            log_return.rolling(int(forward_period))
            .mean()
            .shift(int(-1 - forward_period))
        )
        if forward_period > 1:
            col_name = "forward_return_{}".format(forward_period)
        else:
            col_name = "forward_return"
        forward_return.columns = pd.MultiIndex.from_product(
            [[col_name], forward_return.columns]
        )
        self.targets = forward_return


### Download Data from Yahoo Finance
class YahooAsset(Asset):
    def __init__(
        self, identifiers=["SPY", "QQQ"], startdate="1980-01-01", enddate="2030-12-31"
    ):
        self.identifiers = identifiers
        ## Read price data from Koyfin, discard first trading day as error in adjustments
        column_map = {
            "Adj Close": "adjusted_close",
            "Close": "close",
            "Volume": "volume",
            "Open": "open",
            "High": "high",
            "Low": "low",
        }

        pricedf = yf.download(tickers=identifiers, progress=False)
        pricedf = pricedf.rename(columns=column_map)
        self.price = pricedf.loc[startdate:enddate, :].copy()
        self.price.index = pd.to_datetime(self.price.index)
        ## Default to compute adjusted OHLCV bars
        self.compute_features()

    def compute_features(self):
        # Yahoo volume are already split adjusted
        # Return calculations
        returns = self.price[["adjusted_close"]].pct_change()
        returns.rename(columns={"adjusted_close": "return"}, inplace=True)
        self.price = pd.concat([self.price, returns], axis=1)
        ## Adjusted open calculations
        for column in ["open", "high", "low"]:
            adjusted_open = (
                self.price["adjusted_close"] / self.price["close"] * self.price[column]
            )
            adjusted_open.columns = pd.MultiIndex.from_product(
                [["adjusted_{}".format(column)], adjusted_open.columns]
            )
            self.price = pd.concat([self.price, adjusted_open], axis=1)
        self.price = self.price[
            [
                "return",
                "adjusted_close",
                "adjusted_open",
                "adjusted_high",
                "adjusted_low",
                "volume",
            ]
        ]
        super()._compute_features()
        return self.price


### Loading data downloaded from Compustat and CRSP


def Compustat_CRSP_Data(
    folder="../../data/CRSP",
    linktable="../../data/Compustat_CRSP_link_2021.csv",
    crsptable="../../data/CRSP_names_2021.csv",
    omtable="../../data/Option_Metrics_names_2021.csv",
    numeraitable="../Numerai-Signals/data/numerai_signals_metadata_2021.csv",
    fundamentals_folder="../../data/Compustat_Fundamentals",
    om_folder="../../data/Option_Metrics",
    market_index="Output/CRSP_USCore_all_all_500_index.csv",
    market="USCore",
    sectors=None,
    inds=None,
    subinds=None,
    startyear=1991,
    endyear=2021,
    quantile=5,
    use_fundamentals=False,
    use_option_volume=False,
    use_vol_surface=False,
    debug=False,
    identifier="permno",
):

    pricedf = read_crsp_price(
        folder=folder,
        linktable=linktable,
        crsptable=crsptable,
        omtable=omtable,
        numeraitable=numeraitable,
        market=market,
        sectors=sectors,
        inds=inds,
        subinds=subinds,
        startyear=startyear,
        endyear=endyear,
        debug=debug,
    )
    if pricedf.shape[0] > 0:
        has_price_data = True
    else:
        has_price_data = False

    ### Placeholder
    has_funda_data = False
    has_shortint_data = False
    has_options = False
    fundadf = None
    shortintdf = None
    optionsdf = None

    ## Always read short interests if possible

    selected_gvkeys = pricedf["gvkey"].dropna().unique()

    shortintdf = read_compustat_us_short_interests(
        fundamentals_folder=fundamentals_folder,
        startyear=startyear,
        endyear=endyear,
        selected_gvkeys=selected_gvkeys,
        debug=debug,
    )
    if shortintdf.shape[0] > 0:
        has_shortint_data = True

    if use_fundamentals:

        fundadf = read_compustat_us_fundamentals(
            fundamentals_folder=fundamentals_folder,
            startyear=startyear,
            endyear=endyear,
            selected_gvkeys=selected_gvkeys,
            debug=debug,
        )

        if fundadf.shape[0] > 0:
            has_funda_data = True

    if use_option_volume:

        selected_secids = pricedf["secid"].dropna().unique()

        optionsdf = read_optionmetrics_us_volatility_surface(
            om_folder=om_folder,
            startyear=startyear,
            endyear=endyear,
            use_vol_surface=use_vol_surface,
            selected_secids=selected_secids,
            debug=debug,
        )

        if optionsdf.shape[0] > 0:
            has_options = True

    if has_price_data:
        pivotdf = merge_compustat_dataset(
            pricedf,
            has_fundamentals=has_funda_data,
            fundadf=fundadf,
            has_short_interest=has_shortint_data,
            shortintdf=shortintdf,
            has_options=has_options,
            optionsdf=optionsdf,
            identifier=identifier,
            quantile=quantile,
            debug=debug,
        )
        if pivotdf.shape[0] > 0:
            return pivotdf
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def Compustat_Data(
    folder="../../data/Compustat",
    fundamentals_folder="../../data/Compustat_Fundamentals",
    usnameref="../../data/Compustat_metadata_2020.csv",
    market="US",
    startyear=2016,
    endyear=2021,
    sectors=None,
    inds=None,
    subinds=None,
    quantile=5,
    debug=False,
):

    pricedf = read_compustat_price(
        folder=folder,
        market=market,
        startyear=startyear,
        endyear=endyear,
        sectors=sectors,
        inds=inds,
        subinds=subinds,
        debug=debug,
    )

    ## Compute percentage return
    if pricedf.shape[0] > 0:
        has_pricedata = True
    else:
        has_pricedata = False

    if market == "US":
        fundadf = read_compustat_us_fundamentals(
            fundamentals_folder=fundamentals_folder,
            startyear=startyear,
            endyear=endyear,
            debug=debug,
        )
        ## Use cusip not gvkey as identifier as gvkey is NOT unique for companies with multiple share classes
        identifier = "cusip"
    else:
        ## Not Implemented yet
        # fundadf = read_compustat_international_fundamentals(fundamentals_folder=fundamentals_folder, startyear=startyear, endyear=endyear,)
        fundadf = pd.DataFrame()
        ## Must use isin as identifier since gvkey is NOT unique for companies with multiple share classes
        identifier = "isin"

    if fundadf.shape[0] > 0:
        has_funda_data = True
    else:
        has_funda_data = False

    ### To be Implemented
    ### US Compustat Short Interest
    has_shortint_data = False
    shortintdf = pd.DataFrame()

    if has_pricedata:
        ## Merge
        pivotdf = merge_compustat_dataset(
            pricedf,
            has_funda_data,
            fundadf,
            has_shortint_data,
            shortintdf,
            identifier=identifier,
            quantile=quantile,
            debug=debug,
        )
        if pivotdf.shape[0] > 0:
            return pivotdf
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
