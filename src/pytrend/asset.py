from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Type, Dict, List, Iterable

import pandas as pd
import numpy as np
import os.path

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
