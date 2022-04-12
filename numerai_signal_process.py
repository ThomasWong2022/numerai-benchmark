import pandas as pd
import numpy as np
import datetime
import os
import glob
import gc

from pytrend import Compustat_CRSP_Data


pd.options.mode.chained_assignment = None

if not os.path.exists("features/"):
    os.mkdir("features/")

### Basic
if not os.path.exists("data/temp_basic/"):
    os.mkdir("data/temp_basic/")

### Options
if not os.path.exists("data/temp_options/"):
    os.mkdir("data/temp_options/")

### Financials
if not os.path.exists("data/temp_financials/"):
    os.mkdir("data/temp_financials/")

### Ravenpack Sentiment
if not os.path.exists("data/temp_ravenpack/"):
    os.mkdir("data/temp_ravenpack/")


### targets
if not os.path.exists("data/temp_targets/"):
    os.mkdir("data/temp_targets/")


industry_list = pd.read_csv("../../data/Compustat_industry_code_2021.csv")
industry_list = industry_list[industry_list["gictype"] == "GSECTOR"]


numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv").dropna(
    subset=["hgsubind"]
)
numerai_targets = pd.read_parquet("data/numerai_signals_target_2021.parquet")


def add_industry_labels(CRSP_single_stock, sample_id, sample_id_type="permno"):
    ## Mapped History
    mapped_history = numerai_signals_metadata[
        numerai_signals_metadata[sample_id_type] == sample_id
    ]
    mapped_history["map_start"] = pd.to_datetime(mapped_history["map_start"])
    mapped_history["map_end"] = pd.to_datetime(mapped_history["map_end"])
    mapped_history["crsp_start"] = pd.to_datetime(mapped_history["crsp_start"])
    mapped_history["crsp_end"] = pd.to_datetime(mapped_history["crsp_end"])
    CRSP_single_stock["bloomberg_ticker"] = None
    CRSP_single_stock["group_subindustry"] = None
    for i, row in mapped_history.iterrows():

        if row["map_start"] == datetime.datetime(year=2007, month=4, day=14):
            valid_start = row["crsp_start"]
            valid_end = min(row["crsp_end"], row["map_end"])
        else:
            valid_start = max(row["crsp_start"], row["map_start"])
            valid_end = min(row["crsp_end"], row["map_end"])
        if valid_end > valid_start:
            CRSP_single_stock.loc[valid_start:valid_end, "group_subindustry"] = row[
                "hgsubind"
            ]
            CRSP_single_stock.loc[valid_start:valid_end, "bloomberg_ticker"] = row[
                "bloomberg_ticker"
            ]
    CRSP_single_stock.dropna(
        subset=["bloomberg_ticker", "group_subindustry"], inplace=True
    )
    if CRSP_single_stock.shape[0] > 0:
        ## Derive Group Labels
        CRSP_single_stock["group_subindustry"] = CRSP_single_stock[
            "group_subindustry"
        ].astype(int)
        CRSP_single_stock["group_industry"] = (
            CRSP_single_stock["group_subindustry"] // 100
        )
        CRSP_single_stock["group_sector"] = (
            CRSP_single_stock["group_subindustry"] // 1000000
        )
        ## Downsample to Friday
        shift = pd.to_datetime(CRSP_single_stock.index).dayofweek[0]
        subsampled = (
            CRSP_single_stock.fillna(method="pad")
            .resample("D")
            .fillna(method="pad", limit=31)[11 - shift :: 7]
        )
        subsampled["friday_date"] = subsampled.index.strftime("%Y%m%d").astype(int)
        subsampled["era"] = subsampled.index
        output = subsampled.set_index(["friday_date", "bloomberg_ticker"])
        outputmerged = output.merge(
            numerai_targets[
                [
                    "target_4d",
                    "target_20d",
                ]
            ],
            how="inner",
            left_index=True,
            right_index=True,
        )
        return outputmerged.dropna(
            subset=[
                "target_4d",
                "target_20d",
            ]
        )
    else:
        return pd.DataFrame()


## Feature Transformation per era


def transform_era(df, feature_cols, group_labels=None, keep_original=False):
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


## Sklearn Transformers
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion

from pytrend import SignatureTransformer


### Process Financial Ratios from Open Source AP  (2000 to 2020)
if True:
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    financial_ratios = pd.read_parquet("data/numerai_financials_2021.parquet")
    financial_ratios["rawdatadate"] = pd.to_datetime(
        financial_ratios["yyyymm"], format="%Y%m"
    )

    for sample_id in numerai_signals_metadata["permno"].unique():
        print(sample_id)
        single_stock = financial_ratios[
            financial_ratios["permno"] == sample_id
        ].sort_values("rawdatadate")
        if single_stock.shape[0] > 0:
            ## Data calculated at the end of month can be used for the following month
            single_stock["datadate"] = single_stock["rawdatadate"].shift(-1)
            single_stock.drop(
                [
                    "rawdatadate",
                    "permno",
                    "yyyymm",
                ],
                axis=1,
                inplace=True,
            )
            single_stock.dropna(subset=["datadate"], inplace=True)
            single_stock_daily = (
                single_stock.set_index("datadate").resample("D").asfreq()
            )
            single_stock_daily = single_stock_daily.add_prefix("feature_")
            ans = add_industry_labels(single_stock_daily.copy(), sample_id)
            if ans.shape[0] > 0:
                output = ans[ans["era"] <= "2021-12-31"]
                output.to_parquet(
                    f"data/temp_financials/financials_{sample_id}.parquet"
                )

    del financial_ratios
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_financials/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_financials.parquet"
    )


if True:

    ## Financials
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_financials.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        df_new = df[~df.index.duplicated(keep=False)]
        print(i)
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [col for col in raw_features.columns if col.startswith("group_")]
        normalised.append(
            transform_era(df_new, feature_cols=feature_cols, group_labels=group_labels)
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_financials_normalised.parquet"
    )

## Calculate Basic Factors


class CompustatBasicTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):
        ## Price Based Factors
        log_returns = np.log(1 + X["return"].astype(float))
        market_cap = X["market_cap"].astype(float)

        output_cols = list()

        ## Momentum
        for lookback in [21, 63, 126, 252]:
            output_col = f"momentum_{lookback}"
            X[output_col] = log_returns.rolling(lookback).sum()
            output_cols.append(output_col)

        ## Volatility
        for lookback in [63, 126, 252]:
            output_col = f"volatility_{lookback}"
            X[output_col] = log_returns.rolling(lookback).std() * np.sqrt(
                252 / lookback
            )
            output_cols.append(output_col)

        ## Z-scores
        for lookback in [40, 80, 120, 160, 200]:
            output_col = f"zscore_{lookback}"
            X[output_col] = (
                market_cap - market_cap.rolling(lookback).mean()
            ) / market_cap.rolling(lookback).std()
            output_cols.append(output_col)

        ## Sharpe
        for lookback in [21, 63, 126, 252]:
            output_col = f"sharpe_{lookback}"
            X[output_col] = (
                log_returns.rolling(lookback).mean() / log_returns.rolling(252).std()
            )
            output_cols.append(output_col)

        ## Skewness
        for lookback in [63, 252]:
            output_col = f"skewness_{lookback}"
            X[output_col] = log_returns.rolling(lookback).skew()
            output_cols.append(output_col)

        ## Liquidity,
        for lookback in [63, 252]:
            output_col = f"liquidity_{lookback}"
            X[output_col] = (
                X["dollar_volume"].rolling(lookback).mean() / X["market_cap"]
            )
            output_cols.append(output_col)

        ## Short Interest
        for lookback in [21, 63, 252]:
            output_col = f"shortint_{lookback}"
            X[output_col] = (
                (X["shortint"].fillna(method="pad") * X["close"] / X["market_cap"])
                .rolling(lookback)
                .mean()
            )
            output_cols.append(output_col)

        output = X[output_cols]

        return output.dropna().add_prefix("feature_")


if False:
    CRSP_sample = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=True,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()
    ## Transform for each stock
    for sample_id in permnos:
        print(sample_id)
        try:
            CRSP_single_stock = CRSP_sample.xs(sample_id, level=1, axis=1).dropna(
                subset=["market_cap"]
            )
            transformer = CompustatBasicTransformer()
            output = transformer.transform(CRSP_single_stock.copy())
            ans = add_industry_labels(output.copy(), sample_id)
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_basic/basic_{sample_id}.parquet")
        except:
            print(f"Stock with no data {sample_id}")

    del CRSP_sample
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_basic/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_basic.parquet"
    )

    del ans_list
    gc.collect()


if False:
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_basic.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [col for col in raw_features.columns if col.startswith("group_")]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_basic_normalised.parquet"
    )


## Calculate Targets


class CompustatTargetTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):
        ## Price Based Factors
        log_returns = np.log(1 + X["return"].astype(float))
        market_cap = X["market_cap"].astype(float)

        output_cols = list()

        ## Momentum
        for lookback in [5, 10, 20, 60]:
            output_col = f"return_{lookback}"
            X[output_col] = (
                market_cap.shift(-lookback - 2) - market_cap.shift(-2)
            ) / market_cap.shift(-2)
            output_cols.append(output_col)

        output = X[output_cols]

        return output.dropna().add_prefix("feature_")


if False:
    CRSP_sample = Compustat_CRSP_Data(
        debug=True,
        startyear=2000,
        endyear=2021,
        sectors=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        market="Numerai",
        use_option_volume=False,
        use_vol_surface=False,
        use_fundamentals=True,
        quantile=1,
    )
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    permnos = numerai_signals_metadata["permno"].unique()
    ## Transform for each stock
    for sample_id in permnos:
        print(sample_id)
        try:
            CRSP_single_stock = CRSP_sample.xs(sample_id, level=1, axis=1).dropna(
                subset=["market_cap"]
            )
            transformer = CompustatTargetTransformer()
            output = transformer.transform(CRSP_single_stock.copy())
            ans = add_industry_labels(output.copy(), sample_id)
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_targets/targets_{sample_id}.parquet")
        except:
            print(f"Stock with no data {sample_id}")

    del CRSP_sample
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_targets/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_targets.parquet"
    )

    del ans_list
    gc.collect()


if False:
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_targets.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [col for col in raw_features.columns if col.startswith("group_")]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_targets_normalised.parquet"
    )


### Options


def read_optionmetrics_us_volatility_surface(
    om_folder="../../data/Option_Metrics",
    startyear=2000,
    endyear=2021,
    use_vol_surface=True,
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
                        (sample["days"].isin([30, 60, 91, 182, 365]))
                        & (
                            sample["delta"].isin(
                                [
                                    20,
                                    30,
                                    40,
                                    50,
                                    60,
                                    -20,
                                    -30,
                                    -40,
                                    -50,
                                    -60,
                                ]
                            )
                        )
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

    ## Security
    secprices = list()
    for year in range(startyear, endyear + 1):
        filename = "{}/security_price_{}.parquet".format(om_folder, year)
        if os.path.isfile(filename):
            sample = pd.read_parquet(filename)
            sample = sample[sample["secid"].isin(selected_secids)]
            sample["date"] = pd.to_datetime(sample["date"])
            sample["market_cap"] = sample["close"] * sample["shrout"]
            if debug:
                print("Option Metrics Security Price ", year, sample.shape)
            secprices.append(sample.set_index(["secid", "date"]))
    secprices_merged = pd.concat(secprices, axis=0)

    if use_vol_surface:
        return (
            pd.concat(
                [
                    volsurfaces_merged,
                    optionvolumes_merged,
                    secprices_merged,
                ],
                axis=1,
            )
            .reset_index()
            .dropna()
        )
    else:
        return (
            pd.concat([secprices_merged, optionvolumes_merged], axis=1)
            .reset_index()
            .dropna()
        )


## Compute Implied Volatility Features


class CompustatVolatilityTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):
        ## Volatility Surface
        ## Maturity is 30,60,91,182,365
        ## Delta is +/- 20,30,40,50,60

        output_cols = list()

        ### Average Implied Volatility over
        for lookback in [30.0, 60.0, 91.0]:
            output_col = f"IV_mean_{lookback}"
            iv_lookback_cols = sorted(
                [col for col in X.columns if col.startswith(f"IV_{lookback}")]
            )
            X[output_col] = X[iv_lookback_cols].mean(axis=1)
            output_cols.append(output_col)

        ### Term Structure (Ratio of IV with respect to 1M IV)
        for lookback in [60.0, 91.0]:
            output_col = f"IV_term_{lookback}"
            X[output_col] = X[f"IV_mean_{lookback}"] / X["IV_mean_30.0"]
            output_cols.append(output_col)

        ### Volatility Skew
        for lookback in [
            30.0,
            60.0,
            91.0,
        ]:
            for delta in [
                30.0,
            ]:
                output_col = f"IV_callskew_{lookback}_{delta}"
                X[output_col] = X[f"IV_{lookback}_{delta}"] / X[f"IV_mean_{lookback}"]
                output_cols.append(output_col)
                output_col = f"IV_putskew_{lookback}_-{delta}"
                X[output_col] = X[f"IV_{lookback}_-{delta}"] / X[f"IV_mean_{lookback}"]
                output_cols.append(output_col)

        ### Volatility Skew Term Structure
        for lookback in [
            60.0,
            91.0,
        ]:
            for delta in [
                30.0,
            ]:
                output_col = f"IV_callskew_term_{lookback}_{delta}"
                X[output_col] = (
                    X[f"IV_callskew_{lookback}_{delta}"]
                    / X[f"IV_callskew_30.0_{delta}"]
                )
                output_cols.append(output_col)
                output_col = f"IV_putskew_term_{lookback}_-{delta}"
                X[output_col] = (
                    X[f"IV_putskew_{lookback}_-{delta}"]
                    / X[f"IV_putskew_30.0_-{delta}"]
                )
                output_cols.append(output_col)

        ## Signatures on implied volatility
        output = X[output_cols]
        return output.dropna().add_prefix("feature_")


### Process Option Metrics
if False:
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    selected_secids = numerai_signals_metadata["secid"].unique()
    vol_surface = read_optionmetrics_us_volatility_surface(
        selected_secids=selected_secids
    )
    for sample_id in selected_secids:
        print(sample_id)
        sample = vol_surface[vol_surface["secid"] == sample_id].set_index("date")
        if sample.shape[0] > 0:
            transformer = CompustatVolatilityTransformer()
            output = transformer.transform(sample)
            ans = add_industry_labels(
                output.copy(),
                sample_id,
                sample_id_type="secid",
            )
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_options/options_{sample_id}.parquet")

    del vol_surface
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_options/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_options.parquet"
    )

    del ans_list
    gc.collect()


if False:
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_options.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [col for col in raw_features.columns if col.startswith("group_")]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_options_normalised.parquet"
    )


### Ravenpack
def read_ravenpack_equities(
    ra_folder="../../data/Ravenpack", startyear=2000, endyear=2021, rp_entity_ids=None
):
    ravenpacks = list()
    for year in range(startyear, endyear + 1):
        for month in range(1, 13):
            filename = f"{ra_folder}/Ravenpack_equities_{year}_{month}.parquet"
            ravenpack = pd.read_parquet(filename)
            ravenpack = ravenpack[ravenpack["rp_entity_id"].isin(rp_entity_ids)]
            print(f"Reading Ravepack Equities {year} {month}")
            drop_cols = [
                "headline",
                "rpa_time_utc",
                "timestamp_utc",
                "rp_story_id",
                "product_key",
                "provider_id",
                "provider_story_id",
                "rp_story_event_index",
                "rp_story_event_count",
                "news_type",
                "rp_source_id",
                "source_name",
                "rp_position_id",
                "position_name",
            ]
            ravenpack_small = ravenpack.drop(drop_cols, axis=1)
            ## Filter important events
            ravenpack_important = ravenpack_small[
                (ravenpack_small["event_relevance"] >= 90)
                & (ravenpack_small["event_similarity_days"] >= 1)
                & (ravenpack_small["event_sentiment_score"] != 0)
            ]
            ## Summarise data by event similar keys
            ravenpacks.append(ravenpack_important)
    return pd.concat(ravenpacks, axis=0)


## Get a daily summary of sentiment based on events in the last trading week,month,


class RavenpackSentimentTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def transform(self, X):

        X["rpa_date_utc"] = pd.to_datetime(X["rpa_date_utc"])

        ## Create Daily Index
        daily_newscount = X.groupby("rpa_date_utc")[["event_sentiment_score"]].count()
        daily_newssentiment = X.groupby("rpa_date_utc")[["event_sentiment_score"]].sum()
        daily_css = X.groupby("rpa_date_utc")[["css"]].sum()
        daily_peq = X.groupby("rpa_date_utc")[["peq"]].sum()
        daily_bee = X.groupby("rpa_date_utc")[["bee"]].sum()
        daily_bmq = X.groupby("rpa_date_utc")[["bmq"]].sum()
        daily_bam = X.groupby("rpa_date_utc")[["bam"]].sum()
        daily_bca = X.groupby("rpa_date_utc")[["bca"]].sum()
        daily_ber = X.groupby("rpa_date_utc")[["ber"]].sum()
        daily_anl_chg = X.groupby("rpa_date_utc")[["anl_chg"]].sum()

        dailyX = pd.DataFrame(index=daily_newscount.index)

        output_cols = list()
        ## Event Count
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_EventCount_{lookback}"
            dailyX[output_col] = daily_newscount.rolling(lookback).sum()
            output_cols.append(output_col)
        ## Event Sentiment
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_EventSentiment_{lookback}"
            dailyX[output_col] = daily_newssentiment.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Composite Sentiment
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_Composite_{lookback}"
            dailyX[output_col] = daily_css.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Earnings Evaulation
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_EarningsEval_{lookback}"
            dailyX[output_col] = daily_bee.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Commentary
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_Commentary_{lookback}"
            dailyX[output_col] = daily_bmq.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Mergers
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_Mergers_{lookback}"
            dailyX[output_col] = daily_bam.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Corporate Action
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_CorporateAction_{lookback}"
            dailyX[output_col] = daily_bca.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Earnings Release
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_EarningsRelease_{lookback}"
            dailyX[output_col] = daily_ber.rolling(lookback).mean()
            output_cols.append(output_col)
        ## Analysts Ratings
        for lookback in [
            5,
            10,
            21,
            63,
            252,
        ]:
            output_col = f"rp_AnalystsRatings_{lookback}"
            dailyX[output_col] = daily_anl_chg.rolling(lookback).mean()
            output_cols.append(output_col)

        output = dailyX[output_cols]
        return output.dropna().add_prefix("feature_")


### Process Ravenpack
if False:
    numerai_signals_metadata = pd.read_csv("data/numerai_signals_metadata_2021.csv")
    rp_entity_ids = numerai_signals_metadata["rp_entity_id"].unique()
    ravenpack = read_ravenpack_equities(rp_entity_ids=rp_entity_ids)
    for sample_id in rp_entity_ids:
        print(sample_id)
        sample = ravenpack[ravenpack["rp_entity_id"] == sample_id]
        if sample.shape[0] > 0:
            transformer = RavenpackSentimentTransformer()
            output = transformer.transform(sample)
            ans = add_industry_labels(
                output.copy(),
                sample_id,
                sample_id_type="rp_entity_id",
            )
            if ans.shape[0] > 0:
                ans.to_parquet(f"data/temp_ravenpack/ravenpack_{sample_id}.parquet")

    del ravenpack
    gc.collect()

    ans_list = list()
    files = glob.glob("data/temp_ravenpack/*.parquet")
    for file in files:
        ans = pd.read_parquet(file)
        ans_list.append(ans)
    pd.concat(ans_list, axis=0).to_parquet(
        "data/numerai_signals_features_ravenpack.parquet"
    )

    del ans_list
    gc.collect()

if False:
    raw_features = pd.read_parquet(
        "data/numerai_signals_features_ravenpack.parquet"
    ).sort_values("era")
    normalised = list()
    for i, df in raw_features.groupby("era"):
        print(i)
        df_new = df[~df.index.duplicated(keep=False)]
        feature_cols = [
            col for col in raw_features.columns if col.startswith("feature_")
        ]
        group_labels = [col for col in raw_features.columns if col.startswith("group_")]
        normalised.append(
            transform_era(
                df_new,
                feature_cols=feature_cols,
                group_labels=group_labels,
            )
            .round()
            .astype(int)
        )
    feature_normalised = pd.concat(normalised, axis=0)
    feature_normalised.to_parquet(
        "features/numerai_signals_features_ravenpack_normalised.parquet"
    )
