#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Process Data from Compustat Database
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


# ##### Option Metrics
# # Merging between CRSP and Option Metrics are based on cusip
# ## key field in Option Metrics are secid, key field in CRSP are permno
# ## Use cusip(8-digit) in Option Metrics to link with ncusip in CRSP (which reflects point in time)
# ###


import pandas as pd
import numpy as np
import os.path
import datetime

import requests


#### Read a single year price data from CRSP
def read_crsp_price_data(folder, year):
    ## Mapping CRSP columns
    column_map = {
        "ret": "return",
        "retx": "price_return",
        "shrout": "share_outstanding",
        "openprc": "open",
        "bidlo": "low",
        "askhi": "high",
        "prc": "close",
        "vol": "volume",
        "cfacpr": "price_adjust_factor",
        "cfacshr": "share_adjust_factor",
        "numtrd": "trades",
        "date": "date",
        "cusip": "cusip8",
        "permno": "permno",
        "permco": "permco",
    }

    ## Read One Year of Price Data
    pricefile = "{}/CRSP_{}_price.parquet".format(folder, year)
    if os.path.isfile(pricefile):
        pricedf = pd.read_parquet(pricefile)
    else:
        return None

    pricedf.rename(columns=column_map, inplace=True)
    ## Set Data Types
    pricedf["date"] = pd.to_datetime(pricedf["date"])
    ## Compute Dollar Volume, Market Cap
    pricedf["market_cap"] = pricedf["close"].abs() * pricedf["share_outstanding"] * 1000
    pricedf["dollar_volume"] = pricedf["close"].abs() * pricedf["volume"]
    pricedf["spread"] = np.abs(pricedf["ask"] - pricedf["bid"]) / pricedf["close"].abs()
    ## Split adjust the prices, do not adjust for dividends to avoid looking ahead
    for col in ["close", "open", "high", "low"]:
        pricedf["adjusted_{}".format(col)] = (
            pricedf[col].abs() / pricedf["share_adjust_factor"]
        )

    num_cols = [
        "market_cap",
        "dollar_volume",
        "adjusted_close",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
    ]
    for num_col in num_cols:
        pricedf[num_col] = pricedf[num_col].astype(float)

    ### Pivot
    pricedf.dropna(
        subset=["permno", "date", "dollar_volume", "market_cap"], axis=0, inplace=True
    )
    return pricedf


###  Read CRSP Price Data
###  Merge with Compustat CRSP Link to obtain link to Compustat
###  Merge with CRSP Names Table to obtain share class and start date
###  Merge with Option Metrics Table to obtain link to OM
###  Merge with Numerai Table to obtain link to bloomberg ticker and universe filter
###
def read_crsp_price(
    basefolder="../../data",
    crspfolder="CRSP",
    linktable="Compustat_CRSP_link_2021.csv",
    crsptable="CRSP_names_2021.csv",
    omtable="Option_Metrics_names_2021.csv",
    numeraitable="numerai_signals_metadata_2021.csv",
    market="USCore",
    sectors=None,
    inds=None,
    subinds=None,
    startyear=1921,
    endyear=2021,
    debug=False,
):
    def universe_filter(
        pricedf,
        market="USCore",
        sectors=None,
        inds=None,
        subinds=None,
    ):

        if market == "Numerai":
            numerai_tickers = pd.read_csv(f"{numeraitable}").drop_duplicates(
                subset=["permno", "bloomberg_ticker"]
            )
            numerai_tickers["sector"] = numerai_tickers["hgsubind"] // 1000000
            numerai_tickers["industry"] = numerai_tickers["hgsubind"] // 100
            if sectors:
                numerai_tickers = numerai_tickers[
                    numerai_tickers["sector"].isin(sectors)
                ]
            if inds:
                numerai_tickers = numerai_tickers[
                    numerai_tickers["industry"].isin(inds)
                ]
            pricedf = pricedf[pricedf["permno"].isin(numerai_tickers["permno"])]

        if market == "Commodities":
            commodities = [
                ## Commodities Index
                91129,  ## DBC
                91712,  ## DBA Agriculture
                91715,  ## DBB
                91710,  ## DBP
                91710,  ## DBE
                ## US Bonds
                89468,  ## TLT
                91933,  ## HYG
                89467,  ## LQD
                ## Dollar Index
                91758,  ## UUP
                ## US Market
                85765,  ## DIA
                84398,  ## SPY
                86755,  ## QQQ
                88222,  ## IWM
                ## Stock Sector
                86449,  ## XLB
                86451,  ## XLV
                86452,  ## XLP
                86453,  ## XLY
                86454,  ## XLE
                86455,  ## XLF
                86456,  ## XLI
                86457,  ## XLK
                86458,  ## XLU
                17940,  ## XLC
                15732,  ## XLRE
                ## Commodities Producers
                91232,  ## GDX
                93318,  ## SIL
                93319,  ## COPX
                12372,  ## URA
            ]

            pricedf = pricedf[pricedf["permno"].isin(commodities)]

        return pricedf.copy()

    pricedfs = list()

    output_columns = [
        "cusip",
        "date",
        "close",
        "adjusted_close",
        "adjusted_high",
        "adjusted_open",
        "adjusted_low",
        "spread",
        "return",
        "dollar_volume",
        "market_cap",
        "permno",  ## CRSP Primary Key
        "gvkey",  ## Compustat Foreign Key
        "secid",  ## Option Metrics Foregin Key
        # "cik",  ## SEC EDGAR Foreign Key
        # "tic",  ## Numerai Foregin Key
        # "gind",
        # "gsubind",
        # "sector",
        # "conm",
    ]

    for year in range(startyear, endyear + 1):

        ### Filtering is based on previous year average market cap greater than threshold
        pricedf = read_crsp_price_data(f"{basefolder}/{crspfolder}", year)

        if pricedf is not None:
            ## Merge with Compustat Link Table
            if os.path.isfile(f"{basefolder}/{linktable}"):
                linktabledf = pd.read_csv(f"{basefolder}/{linktable}").dropna(
                    subset=["gvkey", "lpermno", "lpermco"]
                )
                linktabledf.drop_duplicates(
                    subset=[
                        "gvkey",
                        "lpermno",
                        "lpermco",
                    ],
                    inplace=True,
                )
                pricedf = pricedf.merge(
                    linktabledf[
                        [
                            "lpermno",
                            "lpermco",
                            "cik",
                            "gind",
                            "gsubind",
                            "cusip",
                            "conm",
                            "tic",
                            "gvkey",
                        ]
                    ],
                    left_on=["permno", "permco"],
                    right_on=["lpermno", "lpermco"],
                    how="left",
                )

                pricedf.drop(columns=["lpermno", "lpermco"], axis=1, inplace=True)

            ## Merge with Option Metrics Table
            if os.path.isfile(f"{basefolder}/{omtable}"):
                om = (
                    pd.read_csv(f"{basefolder}/{omtable}")
                    .dropna(subset=["ticker"])
                    .drop_duplicates(subset=["cusip", "secid"])
                )
                om["omcusip"] = om["cusip"].copy()
                pricedf = pricedf.merge(
                    om[["omcusip", "secid"]],
                    left_on="cusip8",
                    right_on="omcusip",
                    how="left",
                )

            ## Remove entries without price and trading data
            pricedf.dropna(
                subset=[
                    "market_cap",
                    "dollar_volume",
                    "permno",
                    "cusip",
                ],
                inplace=True,
            )

            pricedf.drop_duplicates(subset=["permno", "cusip", "date"], inplace=True)

            ## Filter down to asset within the required industries and market cap
            pricedf = universe_filter(pricedf, market, sectors, inds, subinds)

            if debug:
                print("CRSP Price ", year, pricedf.shape)

            pricedfs.append(pricedf[output_columns])

    if len(pricedfs) > 0:
        return pd.concat(pricedfs, axis=0, ignore_index=True)
    else:
        return pd.DataFrame()


#### Read Compustat US Fundmentals Data
def read_compustat_us_fundamentals(
    fundamentals_folder="../../data/Compustat_Fundamentals",
    startyear=2016,
    endyear=2021,
    selected_gvkeys=None,
    debug=False,
):

    funda_years = list()

    for datayear in range(startyear - 1, endyear + 1, 1):
        fundafile = "{}/us_fundamentals_quarter_{}.parquet".format(
            fundamentals_folder, datayear
        )
        if os.path.isfile(fundafile):
            funda_year = pd.read_parquet(fundafile)
            funda_year["gvkey"] = funda_year["gvkey"].astype(int)
            funda_year = funda_year[funda_year["gvkey"].isin(selected_gvkeys)]
            funda_year.dropna(
                subset=[
                    "rdq",
                ],
                inplace=True,
            )
            funda_year.sort_values(["gvkey", "rdq"], inplace=True)
            funda_year.drop_duplicates(["gvkey", "rdq"], inplace=True)
            if debug:
                print("Compustat US Fundamentals ", datayear, funda_year.shape)
            funda_years.append(funda_year)

    if len(funda_years) > 0:
        funda_sample = pd.concat(funda_years, axis=0)
        ## Calculating financial data
        funda_sample["buyback"] = funda_sample["cshopq"] * funda_sample["prcraq"]
        funda_sample["dividend"] = funda_sample["dvy"]

        funda_sample["earnings"] = funda_sample["niq"]
        funda_sample["sales"] = funda_sample["saleq"]
        funda_sample["cogs"] = funda_sample["cogsq"]  ## cost of goods sold
        funda_sample["inventory"] = funda_sample["invtq"]
        funda_sample["receivables"] = funda_sample["rectq"]
        funda_sample["income_before_extra_items"] = funda_sample["ibq"]
        funda_sample["pretax_income"] = funda_sample["piq"]
        funda_sample["interest_expense"] = funda_sample["xintq"]

        funda_sample["operating_cashflow"] = funda_sample["oancfy"]
        funda_sample["capital_expenditure"] = funda_sample["capxy"]

        funda_sample["book_value"] = funda_sample["ceqq"]  ## common equity
        funda_sample["intangible_book_value"] = (
            funda_sample["intanq"] - funda_sample["ltq"]
        )
        funda_sample["total_asset"] = funda_sample["atq"]
        funda_sample["total_liability"] = funda_sample["ltq"]
        funda_sample["total_equity"] = funda_sample["teqq"]
        funda_sample["intangible_asset"] = funda_sample["intanq"]
        funda_sample["invested_capital"] = funda_sample["icaptq"]
        funda_sample["longterm_debt"] = funda_sample["dlttq"]
        funda_sample["current_asset"] = funda_sample["actq"]
        funda_sample["current_liability"] = funda_sample["lctq"]
        funda_sample["cash"] = funda_sample["cheq"]  ## cash and short term investment

        ## Financial Report Date
        funda_sample["report_date"] = pd.to_datetime(funda_sample["rdq"])

        ## Fill missing financials with zeros
        financial_cols = [
            "buyback",
            "dividend",
            "earnings",
            "sales",
            "inventory",
            "cogs",
            "receivables",
            "income_before_extra_items",
            "pretax_income",
            "interest_expense",
            "operating_cashflow",
            "capital_expenditure",
            "book_value",
            "intangible_book_value",
            "total_asset",
            "total_liability",
            "total_equity",
            "invested_capital",
            "longterm_debt",
            "current_asset",
            "current_liability",
            "cash",
        ]
        for col in financial_cols:
            funda_sample[col] = funda_sample[col].fillna(0)
        ## Process financial data,
        output_cols = financial_cols + [
            "report_date",
            "gvkey",
        ]
        return funda_sample[output_cols].copy()
    else:
        return pd.DataFrame()


def read_compustat_us_short_interests(
    fundamentals_folder="../../data/Compustat_Fundamentals",
    startyear=2016,
    endyear=2021,
    selected_gvkeys=None,
    debug=False,
):
    shortinterests = list()
    for year in range(startyear, endyear + 1):
        filename = "{}/us_short_interest_{}.parquet".format(
            fundamentals_folder,
            year,
        )
        if os.path.isfile(filename):
            sample = pd.read_parquet(filename)
            sample = sample[sample["gvkey"].isin(selected_gvkeys)]
            sample["date"] = pd.to_datetime(sample["datadate"])
            if debug:
                print("Compustat US Short Int ", year, sample.shape)
            shortinterests.append(
                sample[
                    [
                        "date",
                        "gvkey",
                        "shortint",
                    ]
                ]
            )

    if len(shortinterests) > 0:
        return pd.concat(shortinterests, axis=0)
    else:
        return pd.DataFrame()


# ### Option Metrics Data
# ##
# ##
# ##


def read_optionmetrics_us_volatility_surface(
    om_folder="../../data/Option_Metrics",
    startyear=2016,
    endyear=2021,
    use_vol_surface=False,
    selected_secids=None,
    debug=True,
):
    if use_vol_surface:
        volsurfaces = list()
        for year in range(startyear, endyear + 1):
            for month in range(1, 13):
                filename = "{}/volatility_surface_{}_{}.parquet".format(
                    om_folder, year, month
                )
                if os.path.isfile(filename):
                    sample = pd.read_parquet(filename)
                    ### Filter to include only volatility surface within a year to reduce the number of columns
                    sample = sample[
                        (sample["days"] >= 30)
                        & (sample["days"] <= 252)
                        & (sample["delta"] <= 60)
                        & (sample["delta"] >= -60)
                    ]
                    ## Filter to include data for selected stocks only to reduce memory usage
                    sample = sample[sample["secid"].isin(selected_secids)]
                    ### Make column names
                    sample["date"] = pd.to_datetime(sample["date"])
                    sample["days"] = sample["days"].apply(str)
                    sample["delta"] = sample["delta"].apply(str)
                    volsurface = sample.pivot(
                        index=[
                            "secid",
                            "date",
                        ],
                        columns=["days", "delta"],
                        values="impl_volatility",
                    )
                    volsurface.columns = [
                        "IV_" + "_".join(column).rstrip("_")
                        for column in volsurface.columns.to_flat_index()
                    ]
                    if debug:
                        print(
                            "Option Metrics Volatility Surface ",
                            year,
                            month,
                            volsurface.shape,
                        )
                    volsurfaces.append(volsurface)
        volsurfaces_merged = pd.concat(volsurfaces, axis=0)
    ## Option Volumes
    optionvolumes = list()
    for year in range(startyear, endyear + 1):
        filename = "{}/option_volume_{}.parquet".format(om_folder, year)
        if os.path.isfile(filename):
            sample = pd.read_parquet(filename)
            sample = sample[sample["secid"].isin(selected_secids)]
            sample["date"] = pd.to_datetime(sample["date"])
            sample["cp_flag"].fillna("A", inplace=True)
            optionvolume = sample.pivot(
                index=["secid", "date"],
                columns=["cp_flag"],
                values=["volume", "open_interest"],
            )
            optionvolume.columns = [
                "option_" + "_".join(column).rstrip("_")
                for column in optionvolume.columns.to_flat_index()
            ]
            if debug:
                print("Option Metrics Option Volume ", year, optionvolume.shape)
            optionvolumes.append(optionvolume)
    optionvolumes_merged = pd.concat(optionvolumes, axis=0)
    if use_vol_surface:
        return pd.concat(
            [volsurfaces_merged, optionvolumes_merged], axis=1
        ).reset_index()
    else:
        return optionvolumes_merged.reset_index()


###
### Given price data from CRSP or Compustat, Merge with Fundamentals data from Compustat, Short Interest from Compustat
###
def merge_compustat_dataset(
    pricedf,
    has_fundamentals=False,
    fundadf=None,
    has_short_interest=False,
    shortintdf=None,
    has_options=False,
    optionsdf=None,
    identifier="cusip",
    quantile=5,
    debug=False,
):

    price_cols = list(pricedf.columns)
    selected_cols = list(pricedf.columns)

    if has_fundamentals:
        pricedf = pricedf.merge(
            fundadf,
            left_on=["gvkey", "date"],
            right_on=["gvkey", "report_date"],
            how="left",
        )
        selected_cols.extend(fundadf.columns)
        if debug:
            print("With Fundamentals", pricedf.shape)

    if has_short_interest:
        pricedf = pricedf.merge(
            shortintdf,
            left_on=["gvkey", "date"],
            right_on=["gvkey", "date"],
            how="left",
        )
        selected_cols.extend(shortintdf.columns)
        if debug:
            print("All Short Int ", pricedf.shape)

    ### Add additional Compustat Datasets Here
    ### Compustat Institutioal Holders
    ### Compustat Insiders

    ### Add Option Metrics Data Here
    if has_options:
        pricedf = pricedf.merge(
            optionsdf,
            left_on=["secid", "date"],
            right_on=["secid", "date"],
            how="left",
        )
        selected_cols.extend(optionsdf.columns)
        if debug:
            print("All Options ", pricedf.shape)

    ### pivot dataframe
    selected_cols = list(set(selected_cols) - set([identifier, "date"]))
    pricedf.drop_duplicates(subset=[identifier, "date"], inplace=True)
    ### Filter universe so that we only consider stocks that are shortable and has options
    if has_short_interest:
        pricedf.dropna(
            axis=0,
            subset=[
                identifier,
                "gvkey",
            ],
            inplace=True,
        )
        if debug:
            print("Data with short interest ", pricedf.shape)

    if has_options:
        pricedf.dropna(
            axis=0,
            subset=[
                identifier,
                "secid",
            ],
            inplace=True,
        )
        if debug:
            print("Data with option volumes ", pricedf.shape)

    ### Pivot table into dates index and stocks,features as columns
    pivotdf = pricedf.pivot(index="date", columns=identifier, values=selected_cols)

    ### Create return column for Compustat International price data
    if not "return" in selected_cols and "adjusted_close" in selected_cols:
        ## Forward Fill five days as JPN has missing data in 2003
        returns = pivotdf[["adjusted_close"]].pct_change(fill_method=None)
        returns.rename(columns={"adjusted_close": "return"}, inplace=True)
        pivotdf = pd.concat([pivotdf, returns], axis=1)

    if debug:
        print("Pivoted Data", pivotdf.shape)

    if pivotdf.shape[0] > 0:
        ## Calculate Seasonality Features
        pivotdf["date"] = pivotdf.index.copy()
        pivotdf["year"] = pivotdf["date"].dt.year

        ### Trading days since start of year
        ### Use: Align rebalance dates across years
        ##
        pivotdf["trading_days"] = 0
        prev_year = pivotdf["year"].iloc[0]
        counter = -1
        for i, row in pivotdf[["date", "year"]].iterrows():
            current_year = row["year"][0]
            if current_year == prev_year:
                counter += 1
            else:
                counter = 0
            pivotdf.loc[i, "trading_days"] = counter
            prev_year = current_year

    return pivotdf


### Read Compustat Price Data and crate market cap based filters
###
###
def read_compustat_price_year(folder, market, year):

    ## Read price data from Compustat
    column_map = {
        "datadate": "date",
        "trfd": "total_return_factor",
        "ajexdi": "adjust_ratio",
        "cshoc": "share_outstanding",
        "cshtrd": "volume",
        "prccd": "close",
        "prcld": "open",
        "prchd": "high",
        "prcod": "low",
        "curcdd": "currency",
        "exchg": "exchange",
        "cusip": "cusip",
        "isin": "isin",
        "gsubind": "gsubind",
        "gind": "gind",
    }

    ### Read Previous Year Pircedata and create investable universe for next year based on market cap

    if market == "US":
        pricefilename = "{}_{}/{}_price_{}.parquet".format(folder, market, market, year)
        identifier = "cusip"
    else:
        pricefilename = "{}_International/{}_price_{}.parquet".format(
            folder, market, year
        )
        identifier = "isin"
    if os.path.isfile(pricefilename):
        pricedf = pd.read_parquet(pricefilename)
        if identifier in pricedf.columns:
            pricedf.set_index(["isin", "datadate"], inplace=True)
        pricedf.reset_index(inplace=True)
    else:
        return None, None

    pricedf.rename(columns=column_map, inplace=True)
    pricedf["date"] = pd.to_datetime(pricedf["date"])
    ## Compute Dollar Volume, Market Cap, Adjusted Price
    pricedf["market_cap"] = pricedf["close"] * pricedf["share_outstanding"]
    pricedf["dollar_volume"] = pricedf["close"] * pricedf["volume"]
    for col in ["close", "open", "high", "low"]:
        pricedf["adjusted_{}".format(col)] = (
            pricedf[col] / pricedf["adjust_ratio"] * pricedf["total_return_factor"]
        )

    pricedf.sort_values(
        [
            "date",
            "gvkey",
            "adjusted_close",
        ],
        inplace=True,
    )
    pricedf.drop_duplicates(subset=["date", "gvkey"], inplace=True)
    pricedf.dropna(
        subset=[
            "date",
            "gvkey",
            "market_cap",
            "dollar_volume",
        ],
        inplace=True,
        axis=0,
    )

    return pricedf


# ## Loading data downloaded from Compustat and CRSP


def Compustat_CRSP_Data(
    basefolder="../../data",
    crspfolder="CRSP",
    linktable="Compustat_CRSP_link_2021.csv",
    crsptable="CRSP_names_2021.csv",
    omtable="Option_Metrics_names_2021.csv",
    numeraitable="numerai_signals_metadata_2021.csv",
    fundamentals_folder="Compustat_Fundamentals",
    om_folder="Option_Metrics",
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
        basefolder=basefolder,
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
        fundamentals_folder=f"{basefolder}/{fundamentals_folder}",
        startyear=startyear,
        endyear=endyear,
        selected_gvkeys=selected_gvkeys,
        debug=debug,
    )
    if shortintdf.shape[0] > 0:
        has_shortint_data = True

    if use_fundamentals:

        fundadf = read_compustat_us_fundamentals(
            fundamentals_folder=f"{basefolder}/{fundamentals_folder}",
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
            om_folder=f"{basefolder}/{om_folder}",
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
