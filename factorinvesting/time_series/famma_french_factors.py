# cursor_test.py
# MainSequence TimeSeries: Fama-French 3-Factor Model Example
# This script demonstrates how to build a time series pipeline for the Fama-French 3-factor model using the mainsequence SDK.

from mainsequence.tdag import TimeSerie, APITimeSerie
import mainsequence.client as ms_client

import datetime
import pytz
import pandas as pd
import requests
import io
import numpy as np
from mainsequence.virtualfundbuilder.contrib.prices.time_series import get_interpolated_prices_timeseries
from mainsequence.tdag.time_series import ModelList
from mainsequence.virtualfundbuilder.models import AssetsConfiguration, PricesConfiguration
from mainsequence.virtualfundbuilder.enums import PriceTypeNames
from typing import Union
from tqdm import tqdm
# URL for Fama-French 3 Factors daily data (CSV format)
FRENCH_FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

_ff3_cache = None  # Module-level cache to avoid repeated downloads

def fetch_ff3_factors():
    """
    Download and parse the Fama-French 3-factor daily data from the French website.
    Returns a DataFrame indexed by datetime (UTC), columns: ['MKT', 'SMB', 'HML'] (in percent, e.g. 0.12 for 12bp).
    """
    global _ff3_cache
    if _ff3_cache is not None:
        return _ff3_cache
    # Download and unzip
    resp = requests.get(FRENCH_FF3_URL)
    resp.raise_for_status()
    import zipfile
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.endswith('.CSV') or name.endswith('.csv'):
                with zf.open(name) as f:
                    lines = f.read().decode('utf-8').splitlines()
                    # Find the header line (starts with ',Mkt-RF,SMB,HML,RF')
                    header_idx = None
                    for i, line in enumerate(lines):
                        if line.strip().startswith(',Mkt-RF'):
                            header_idx = i
                            break
                    if header_idx is None:
                        raise RuntimeError("Could not find Fama-French CSV header line.")
                    # Only keep rows where the first column is 8 digits (date)
                    data_lines = [lines[header_idx]]
                    for row in lines[header_idx+1:]:
                        first_col = row.split(',')[0].strip()
                        if len(first_col) == 8 and first_col.isdigit():
                            data_lines.append(row)
                        else:
                            break  # Stop at first non-date row (footer)
                    data = '\n'.join(data_lines)
                    try:
                        df = pd.read_csv(io.StringIO(data))
                        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
                        df.set_index('Date', inplace=True)
                        df.rename(columns={'Mkt-RF': 'MKT', 'SMB': 'SMB', 'HML': 'HML'}, inplace=True)
                        for col in ['MKT', 'SMB', 'HML']:
                            df[col] = df[col] / 100.0
                        _ff3_cache = df[['RF','MKT', 'SMB', 'HML']]
                        return _ff3_cache
                    except Exception as e:
                        raise RuntimeError(f"Error parsing Fama-French CSV: {e}")
    raise RuntimeError('Could not find Fama-French CSV in zip file')

class FamaFrench3FactorTimeSerie(TimeSerie):
    """
    MainSequence TimeSerie for the Fama-French 3-factor model.

    This class demonstrates how to build a robust, historical, non-asset time series for use in the MainSequence platform.
    It fetches the Fama-French 3-factor data (MKT, SMB, HML) from the Kenneth French Data Library,
    processes it into a DataFrame with a UTC datetime index, and provides detailed metadata for each column.

    Key features and best practices illustrated here:
    - Full historical incremental update logic: only new data since the last observation is returned.
    - No asset logic: this time series is for macro factors, not asset-level data.
    - Proper time_index handling: always timezone-aware (UTC) and named 'time_index'.
    - Lowercase column names for platform compatibility.
    - Rich column metadata for downstream interpretability.
    - Canonical _run_post_update_routines for registering the time series in the platform with a clear name and description.
    - Can be used as a template for other macro, economic, or factor time series.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the FamaFrench3FactorTimeSerie.
        No dependencies are required for this macro factor time series.
        """
        super().__init__(*args, **kwargs)

    def update(self, update_statistics):
        """
        Incrementally update the time series with new Fama-French 3-factor data.

        Args:
            update_statistics: UpdateStatistics object from MainSequence, used to determine the last observation.

        Returns:
            pd.DataFrame: New rows of factor data since the last observation, indexed by 'time_index' (datetime64[ns, UTC]),
            with columns ['mkt', 'smb', 'hml'] (all lowercase, float).

        Notes:
            - The DataFrame index is always timezone-aware UTC for platform compatibility.
            - If there is no new data, returns an empty DataFrame.
            - This logic can be adapted for any macro/factor time series with similar update requirements.
        """
        # Get the last observation date from update_statistics
        last_observation = update_statistics.get_max_latest_value()

        if datetime.datetime.now(pytz.utc)-datetime.timedelta(days=90) <= (last_observation or datetime.datetime(year=1900,month=1,day=1,tzinfo=pytz.utc)):
            return pd.DataFrame() #does not have API and guarantee not update

        ff3 = fetch_ff3_factors()
        # If there is a last observation, only return data after that date
        if last_observation is not None:
            mask = ff3.index > pd.Timestamp(last_observation.date())
            new_data = ff3.loc[mask]
        else:
            new_data = ff3
        # If nothing new, return empty DataFrame
        if new_data.empty:
            return pd.DataFrame()
        # Set index name to 'time_index' for MainSequence compatibility
        new_data.index.name = 'time_index'
        # Force index to be datetime64[ns, UTC] robustly
        new_data.index = pd.to_datetime(new_data.index, utc=True)
        # Ensure columns are lower case for platform compatibility
        new_data = new_data.rename(columns={"MKT": "mkt", "SMB": "smb", "HML": "hml","RF":"rf"})
        return new_data[["mkt", "smb", "hml","rf"]]

    def _run_post_update_routines(self, error_on_last_update, update_statistics):
        """
        Register this time series in the MainSequence platform with a clear name and description.
        This is important for discoverability and downstream use.
        No assets are added, as this is a macro/factor time series.

        Best practices:
        - Use a unique, descriptive identifier (e.g., 'fama_french_3factor').
        - Provide a clear description of the data and its source.
        - This pattern can be reused for any non-asset time series.
        """
        MARKET_TIME_SERIES_UNIQUE_IDENTIFIER = "fama_french_3factor"
        DESCRIPTION = (
            "This time series contains the Fama-French 3-factor model (MKT, SMB, HML) "
            "downloaded from the Kenneth French Data Library."
        )
        try:
            markets_time_series_details = ms_client.MarketsTimeSeriesDetails.get(
                unique_identifier=MARKET_TIME_SERIES_UNIQUE_IDENTIFIER,
            )
            if markets_time_series_details.related_local_time_serie.id != self.local_time_serie.id:
                markets_time_series_details = markets_time_series_details.patch(related_local_time_serie__id=self.local_time_serie.id)
        except ms_client.DoesNotExist:
            markets_time_series_details = ms_client.MarketsTimeSeriesDetails.update_or_create(
                unique_identifier=MARKET_TIME_SERIES_UNIQUE_IDENTIFIER,
                related_local_time_serie__id=self.local_time_serie.id,
                data_frequency_id=ms_client.DataFrequency.one_d,
                description=DESCRIPTION,
            )
        # No asset logic, as this time series does not have assets

    def get_column_metadata(self):
        """
        Provide rich metadata for each column in the time series.
        This improves interpretability and downstream usage in the MainSequence platform.

        Returns:
            List[ColumnMetaData]: Metadata for each factor column.

        Best practices:
        - Use clear, human-readable labels and descriptions.
        - Specify dtype for each column.
        - This pattern can be reused for any time series with well-defined columns.
        """
        from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [
            ColumnMetaData(
                column_name="mkt",
                dtype="float",
                label="Market Excess Return (MKT)",
                description="Fama-French market excess return factor (MKT), from the Kenneth French Data Library."
            ),
            ColumnMetaData(
                column_name="smb",
                dtype="float",
                label="Small Minus Big (SMB)",
                description="Fama-French SMB factor: return of small minus big stocks, from the Kenneth French Data Library."
            ),
            ColumnMetaData(
                column_name="hml",
                dtype="float",
                label="High Minus Low (HML)",
                description="Fama-French HML factor: return of high minus low book-to-market stocks, from the Kenneth French Data Library."
            ),
        ]
        return columns_metadata

class ThreeFFLoadingsTimeSerie(TimeSerie):
    """
    MainSequence TimeSerie for rolling Fama-French 3-factor loadings (betas) for a list of assets.

    This class computes rolling regression loadings (betas) of asset returns on the Fama-French 3 factors (MKT, SMB, HML)
    for a given asset list and rolling window. It uses get_interpolated_prices_timeseries to fetch asset price data and
    fetches FamaFrench3FactorTimeSerie data as a dependency using get_df_between_dates.

    Inputs:
        - rolling_window: int, the window size (in days) for the rolling regression.
        - asset_list: ModelList of asset objects (ignored as dependency, but used for price fetching).
        - local_kwargs_to_ignore: list of str, kwargs to ignore as dependencies (default: ["asset_list"])

    Output:
        - DataFrame indexed by ['time_index', 'unique_identifier'], columns: ['mkt_loading', 'smb_loading', 'hml_loading']
          (all lowercase, float), with time_index as datetime64[ns, UTC].

    Best practices:
        - asset_list is added to local_kwargs_to_ignore so it is not treated as a dependency.
        - All columns and index names are platform-compatible.
        - Detailed docstrings and comments for clarity and reusability.
    """

    def __init__(self,  assets_category_unique_id: str,rolling_window: int = 60, local_kwargs_to_ignore = ["assets_category_unique_id"], *args, **kwargs):
        """
        Initialize the ThreeFFLoadingsTimeSerie.
        Args:
            assets_category_unique_id (str): Unique identifier for the asset category to compute loadings for.
            rolling_window (int): Window size for rolling regression (default: 60 days).
            local_kwargs_to_ignore (list): List of kwarg names to ignore as dependencies (default: ["assets_category_unique_id"])
        """
        
        self.rolling_window = rolling_window
        prices_config = PricesConfiguration(bar_frequency_id="1d")
        self.assets_category_unique_id=assets_category_unique_id
        assets_configuration = AssetsConfiguration(
                        assets_category_unique_id="s&p500_constitutents",
                        price_type=PriceTypeNames.CLOSE,
                        prices_configuration=prices_config
                    )

        self.prices_ts = get_interpolated_prices_timeseries(assets_configuration)

        self.factors_ts = FamaFrench3FactorTimeSerie()
        super().__init__(*args, local_kwargs_to_ignore=local_kwargs_to_ignore, **kwargs)


    def get_asset_list(self) ->Union[None, list]:
        """
        Returns the list of assets to be included in the time series calculations.

        This method overrides the default asset selection logic by filtering and returning only the assets
        specified by the assets_category_unique_id associated with this TimeSerie. This allows update_statistics
        and other methods to operate only on the requested subset of assets, rather than the full asset universe.
        The returned list is wrapped in a ModelList for compatibility with MainSequence asset-based workflows.

        Returns:
            ModelList: List of asset objects filtered by the specified asset category.
        """
        # Fetch S&P500 asset category
        pool_asset_category = ms_client.AssetCategory.get(unique_identifier=self.assets_category_unique_id)
        # Filter assets in the category
        assets = ms_client.Asset.filter(id__in=pool_asset_category.assets)
        # Wrap in ModelList
        asset_list = ModelList(assets)
        return asset_list
    
    def update(self, update_statistics):
        """
        Compute rolling Fama-French 3-factor loadings (betas) for each asset in the dynamically fetched asset universe.

        This method demonstrates the canonical MainSequence pattern for asset-based time series with dependencies:
        - The asset universe is dynamically fetched using self.get_asset_list(), allowing for flexible universes.
        - Price data is fetched using self.prices_ts, which is configured in the constructor.
        - Factor data is fetched using self.factors_ts, a dependency time series.
        - Rolling OLS regressions are computed for each asset over the specified window.
        - The output DataFrame is indexed by ['time_index', 'unique_identifier'] and columns are lower case and platform-compatible.
        - The DataFrame is filtered using update_statistics to ensure only new/needed data is returned.

        This template can be adapted for any asset-based time series that needs to compute rolling statistics or loadings
        using one or more dependency time series.

        Args:
            update_statistics: UpdateStatistics object from MainSequence, used to determine the last observation and filter output.

        Returns:
            pd.DataFrame: Rolling factor loadings for each asset, indexed by ['time_index', 'unique_identifier'],
            columns: ['mkt_loading', 'smb_loading', 'hml_loading'] (all lowercase, float).
        """
        # Dynamically fetch the asset universe
        asset_list = self.get_asset_list()
        if not asset_list:
            return pd.DataFrame()

        start_date = None

        raw_prices = self.prices_ts.get_df_between_dates(
            unique_identifier_range_map=update_statistics.get_update_range_map_great_or_equal()

        )

        # Fetch Fama-French factors using dependency time series
        ff3 = self.factors_ts.get_df_between_dates(
            start_date=start_date,
            great_or_equal=True
        )

        if ff3.empty or raw_prices.empty:
            return pd.DataFrame()

        index_names=raw_prices.index.names
        raw_prices=raw_prices.reset_index()
        raw_prices["time_index"]=raw_prices["time_index"].dt.normalize()
        raw_prices=raw_prices.set_index(index_names)


        original_index=raw_prices.index
        results = []
        for asset in tqdm(asset_list,desc="building_rolling_regreesion"):
            asset_uid = asset.unique_identifier
            asset_prices = raw_prices[raw_prices.index.get_level_values("unique_identifier") == asset_uid][["close"]]
            asset_returns = asset_prices.pct_change().dropna()
            # Align with factors
            aligned = (
                asset_returns
                .join(ff3[["mkt", "smb", "hml"]], how="inner")
                .dropna()
            )
            if len(aligned) < self.rolling_window:
                continue
            # Rolling regression for each window
            for i in range(self.rolling_window, len(aligned)+1):
                window = aligned.iloc[i-self.rolling_window:i]
                X = window[["mkt", "smb", "hml"]]
                y = window["close"]-window["rf"] #excess return
                X_ = X.copy()
                X_["const"] = 1.0
                try:
                    coefs = pd.Series(np.linalg.lstsq(X_, y, rcond=None)[0], index=["mkt", "smb", "hml", "const"])
                except Exception:
                    continue
                results.append({
                    "time_index":window.index[-1][0],
                    "unique_identifier": asset_uid,
                    "mkt_loading": coefs["mkt"],
                    "smb_loading": coefs["smb"],
                    "hml_loading": coefs["hml"]
                })
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["time_index"] = pd.to_datetime(df["time_index"], utc=True)
        df.set_index(["time_index", "unique_identifier"], inplace=True)

        return df

# Example usage
if __name__ == "__main__":
    # Example usage for running rolling Fama-French 3-factor loadings for a given asset universe
    # The asset universe is specified by assets_category_unique_id (default: 's&p500_constitutents' for S&P500)
    # To use a different universe, change the assets_category_unique_id argument below
    assets_category_unique_id = 's&p500_constitutents'
    rolling_window = 252 * 4  # 4 years of daily data
    ts = ThreeFFLoadingsTimeSerie(assets_category_unique_id=assets_category_unique_id, rolling_window=rolling_window)
    ts.run(debug_mode=True, force_update=True) 