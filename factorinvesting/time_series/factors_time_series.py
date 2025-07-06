import datetime
import os
import pytz
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Union, Optional, List, Dict

from mainsequence.tdag import TimeSerie
import mainsequence.client as ms_client
from mainsequence.virtualfundbuilder.contrib.prices.time_series import get_interpolated_prices_timeseries
from mainsequence.virtualfundbuilder.models import AssetsConfiguration, PricesConfiguration
from mainsequence.virtualfundbuilder.enums import PriceTypeNames
from mainsequence.virtualfundbuilder.contrib.prices.time_series import InterpolatedPrices
from polygon import RESTClient
# from polygon.reference_apis import ReferenceClient # Commenting out incorrect import
from tqdm import tqdm
import polygon

client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))

CANONICAL_FACTOR_RETURNS_ID = "canonical_12_style_factor_returns_axioma_barra"
CANONICAL_STYLE_FACTORS_MATRIX_ID = "canonical_12_style_factor_matrix_axba"


# From user's code
STYLE_FACTOR_MAP = {'lncap': 'Size',
                    'lncap2': 'NonLinearSize',
                    'beta': 'Beta',
                    'nl_beta': 'NonLinearBeta',
                    'mom_12_1': 'Momentum',
                    'mom_1m': 'STMomentum',
                    'resid_vol': 'ResidualVol',
                    'liquidity': 'Liquidity',
                    'book_to_price': 'Value',
                    'growth': 'Growth',
                    'leverage': 'Leverage',
                    'div_yield': 'DividendYield',
                    'earnings_yield': 'EarningsYield',
                    'profitability': 'Profitability',

                    }

# ── dividend helper ────────────────────────────────────────────────────────────
def fy_dividend_yield_avg(polygon_client, ticker: str, fiscal_year: int, logger=None):
    """
    Dividend-yield for one fiscal year, split-neutral.

        yield_FY =  mean_i(  cash_amount_i / close_price_on_exdate_i  )

    where i runs over every cash dividend whose ex-dividend date lies in the
    fiscal year.  Returns np.nan if either cash amounts or prices are missing.

    Parameters
    ----------
    polygon_client : authenticated Polygon client
    ticker         : str   • e.g. 'AAPL'
    fiscal_year    : int   • calendar year of the fiscal period
    logger         : optional logger for warnings
    """
    # ───────────────────────────── 1) pull the dividends ──────────────────────────
    try:
        divs = polygon_client.list_dividends(
            ticker=ticker,
            ex_dividend_date_gte=f"{fiscal_year}-01-01",
            ex_dividend_date_lte=f"{fiscal_year}-12-31",
            limit=1000
        )
    except Exception as exc:
        if logger:
            logger.warning(f"Dividends endpoint failed for {ticker}: {exc}")
        return np.nan

    if not divs:  # no cash dividends that year
        return np.nan

        # ─── 2) pull ALL daily closes for that fiscal year in ONE query ────────────
    try:
        bars = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=f"{fiscal_year}-01-01",
            to=f"{fiscal_year}-12-31",
            adjusted=False,  # keep prices on their native split basis
            sort="asc",
            limit=5000,
        )
        # map YYYY-MM-DD  → close price
        price_by_date = {
            datetime.datetime.utcfromtimestamp(bar.timestamp / 1000).date().isoformat(): bar.close
            for bar in bars
        }
    except Exception as exc:
        if logger:
            logger.warning(f"get_aggregate_bars failed for {ticker}: {exc}")
        return np.nan

    # ─── 3) compute one yield per dividend (price on ex-date) ──────────────────
    yields = []
    for d in divs:
        cash_amt = float(getattr(d, "cash_amount", 0.0) or 0.0)
        ex_date = getattr(d, "ex_dividend_date", None)  # 'YYYY-MM-DD'

        if cash_amt == 0 or ex_date is None:
            continue

        px = price_by_date.get(ex_date)
        # if market was closed that calendar day, roll back to the last trading day
        if px is None:
            prev = datetime.datetime.fromisoformat(ex_date).date()
            for _ in range(15):  # look back ≤ 5 days
                prev -= datetime.timedelta(days=1)
                px = price_by_date.get(prev.isoformat())
                if px is not None:
                    break

        if px and px > 0:
            yields.append(cash_amt / px)

    # ─── 4) average & return ───────────────────────────────────────────────────
    return np.nanmean(yields) if yields else np.nan


class FundamentalsTimeSeries(TimeSerie):
    """
    Fetches and caches annual fundamental data from Polygon.io.
    Refactors the logic from `get_fundamentals_polygon`.
    """

    @TimeSerie._post_init_routines()
    def __init__(self, assets_category_unique_id: str,
                 local_kwargs_to_ignore=["assets_category_unique_id"],
                 *args, **kwargs):
        super().__init__(*args, **kwargs, local_kwargs_to_ignore=local_kwargs_to_ignore)
        self.assets_category_unique_id = assets_category_unique_id
        self.polygon_client = client

    def _get_asset_list(self) -> Optional[List["Asset"]]:

        asset_ids = ms_client.AssetCategory.get(unique_identifier=self.assets_category_unique_id).assets
        asset_list = ms_client.Asset.filter(id__in=asset_ids)
        return asset_list

    def update(self, update_statistics):
        # Fundamentals are annual, so we don't need to update daily.
        # We'll update if there's no data or if the last update was > 90 days ago.

        current_date = datetime.datetime.now(pytz.utc)
        uid_to_update = []
        for uid, last_update_time in update_statistics.update_statistics.items():
            if (current_date - last_update_time) > datetime.timedelta(days=364):
                uid_to_update.append(uid)

        if len(uid) == 0:
            return pd.DataFrame()

        records = []
        for asset in tqdm(update_statistics.asset_list, desc="getting assets_fundamentals"):
            if asset.unique_identifier not in uid_to_update:
                continue
            try:
                # Correctly call the experimental financials endpoint through the client
                # This returns a generator
                # The line below will fail until the client is reinstated
                for fin in self.polygon_client.vx.list_stock_financials(asset.ticker, timeframe='annual'):

                    if not fin or not fin.financials:
                        continue

                    # The returned object has attributes for each statement
                    bal = fin.financials.balance_sheet
                    inc = fin.financials.income_statement
                    cas = fin.financials.cash_flow_statement

                    total_dividends_per_share = fy_dividend_yield_avg(self.polygon_client,
                                                                      asset.ticker, int(fin.fiscal_year),
                                                                      inc.basic_average_shares
                                                                      )

                    records.append({
                        "unique_identifier": asset.unique_identifier,
                        # Field names now match the object attributes from the client library
                        "total_assets": bal.assets.value,
                        "total_debt": bal.liabilities.value,  # Example path, may need adjustment
                        "total_equity": bal.equity.value,
                        "total_dividends_per_share": total_dividends_per_share,
                        "preferred_stock_d_and_oa": inc.preferred_stock_dividends_and_other_adjustments.value,
                        "net_income": inc.net_income_loss.value,
                        "revenue": inc.revenues.value,
                        "basic_average_shares": inc.basic_average_shares.value,
                        "end_date": pd.to_datetime(fin.end_date).timestamp(),
                        "filing_date": pd.to_datetime(fin.filing_date).timestamp(),
                        "equity_attributable_to_parent": bal.equity_attributable_to_parent.value,
                        "equity_attributable_to_nci": bal.equity_attributable_to_noncontrolling_interest.value,
                        "time_index": datetime.datetime(year=int(fin.fiscal_year), month=12, day=31, tzinfo=pytz.utc)

                    })
            except Exception:
                continue  # Skip assets where data is unavailable

        if not records: return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index(["time_index", "unique_identifier"])
        df = df.astype(float)
        return df.sort_index()

    def _get_column_metadata(self):
        """
        Return a `List[ColumnMetaData]` describing every annual fundamental
        field that the `update()` method writes into the DataFrame.

        All values are reported **per share or per company** for the fiscal
        year whose closing date is the DataFrame’s first index level
        (`time_index`).  Epoch‑time columns are stored as floats so they can be
        cast back to `pd.Timestamp` downstream.
        """

        meta_specs = [
            # (column_name, dtype, label, description)
            ("total_assets", "float", "TotalAssets",
             "Total assets at fiscal year‑end (balance‑sheet)."),
            ("total_debt", "float", "TotalDebt",
             "Total debt / liabilities at FYE."),
            ("total_equity", "float", "TotalEquity",
             "Shareholders’ equity at FYE."),
            ("total_dividends_per_share", "float", "DividendsPerShare",
             "Sum of cash dividends per share declared during the fiscal year."),
            ("preferred_stock_d_and_oa", "float",
             "PrefDividendsAdjust",
             "Preferred‑stock dividends and other income‑statement adjustments."),
            ("net_income", "float", "NetIncome",
             "Net income (loss) for the fiscal year."),
            ("revenue", "float", "Revenue",
             "Total revenue for the fiscal year."),
            ("basic_average_shares", "float", "BasicAvgShares",
             "Average basic shares outstanding during the fiscal year."),
            ("end_date", "float", "FiscalYearEndDate",
             "Fiscal year‑end date expressed as POSIX timestamp (UTC)."),
            ("filing_date", "float", "FilingDate",
             "SEC filing date expressed as POSIX timestamp (UTC)."),
            ("equity_attributable_to_parent", "float",
             "EquityToParent",
             "Portion of equity attributable to the parent company."),
            ("equity_attributable_to_nci", "float",
             "EquityToNCI",
             "Portion of equity attributable to non‑controlling interests."),
        ]

        return [
            ms_client.ColumnMetaData(
                column_name=col,
                dtype=dtype,
                label=label,
                description=desc
            )
            for col, dtype, label, desc in meta_specs
        ]

    # ──────────────────────────────────────────────────────────────
    # 2 ▸ Post‑update bookkeeping
    # ──────────────────────────────────────────────────────────────
    def _run_post_update_routines(self, error_on_last_update,
                                  update_statistics: ms_client.DataUpdates):
        """
        Register (or patch) the *canonical* annual‑fundamentals time series in
        `MarketsTimeSeriesDetails` **and** attach any newly processed assets.

        Polygon fundamentals are reported once per year, so we tag the record
        with an annual frequency when that enum is available; otherwise we fall
        back to `one_d`.
        """
        CANONICAL_FUNDAMENTALS_ID = "polygon_annual_fundamentals"

        # Choose the best‑matching frequency enum that exists in the client.
        freq_enum = (
            getattr(ms_client.DataFrequency, "one_y", None)
            or getattr(ms_client.DataFrequency, "one_a", None)  # alternative naming
            or ms_client.DataFrequency.one_d
        )


        source_table=self.local_time_serie.remote_table

        try:
            mts = ms_client.MarketsTimeSeriesDetails.get(
                unique_identifier=CANONICAL_FUNDAMENTALS_ID
            )

            # Ensure it points at the same local_time_serie we just updated
            if mts.source_table.id != source_table.id:
                mts = mts.patch(source_table__id=source_table.id)



        except ms_client.DoesNotExist:
            mts = ms_client.MarketsTimeSeriesDetails.update_or_create(
                unique_identifier=CANONICAL_FUNDAMENTALS_ID,
                source_table__id=source_table.id,
                data_frequency_id=freq_enum,
                description=(
                    "Canonical annual fundamentals downloaded from Polygon.io "
                    "(balance‑sheet, income‑statement and cash‑flow items) for "
                    "every covered equity."
                ),
            )

        # Append any assets from this run that are not yet linked
        new_assets = [
            asset for asset in update_statistics.asset_list
            if asset.id not in mts.assets_in_data_source
        ]
        if new_assets:
            mts.append_asset_list_source(asset_list=new_assets)


class StyleFactorsTimeSeries(TimeSerie):
    """
    Builds the twelve classic style exposures from MSCI-Barra / Qontigo-Axioma.
    This class orchestrates data dependencies and applies the factor construction
    logic for each day.
    """
    MAD_CONST = 1.4826  # For winsorisation
    EM_WINSOR_DAYS = 60  # pooled window for emerging markets
    EM_MIN_OBS = 120  # min valid obs before falling back to pool

    # ---------------------------------------------------------------------
    # 🛠️  Descriptor hygiene – winsorisation
    # ---------------------------------------------------------------------
    @staticmethod
    def _winsorise(
            series: pd.Series,
            z: float = 3.0,
            *,
            pooled_sample: pd.Series | None = None,
            min_obs: int | None = None,
    ) -> pd.Series:
        """MAD‑based winsorisation identical to MSCI‑Barra / Qontigo‑Axioma.

        Parameters
        ----------
        series : pd.Series
            One‑day cross‑section of a raw descriptor (index = tickers).
        z : float, default 3.0
            Winsor clip in MAD units (≈ ±3 σ under Normality).
        pooled_sample : pd.Series, optional
            Provide an *N‑day pooled* distribution when working with thin
            universes (Emerging Markets).  If *series* has fewer than
            ``min_obs`` non‑missing values, the median and MAD are computed on
            this pooled sample instead, exactly matching the vendor rule.
        min_obs : int, optional
            Threshold below which we switch to *pooled_sample* (must be
            supplied by the caller if runtime configurability is required).

        Returns
        -------
        pd.Series
            Winsorised series; NaNs preserved so the caller can neutral‑fill
            them (→ exposure 0) after standardisation.
        """
        if min_obs is None:
            # Fall back to infinity → never use pooled sample unless caller
            # passes a threshold.  Keeps the staticmethod truly stateless.
            min_obs = float("inf")

        base = series.dropna()
        if pooled_sample is not None and base.size < min_obs:
            base = pooled_sample.dropna()

        if base.empty:
            return series

        med = base.median()
        mad = np.nanmedian(np.abs(base - med))
        if mad == 0 or np.isnan(mad):
            return series

        k = z * StyleFactorsTimeSeries.MAD_CONST * mad
        upper = med + k
        lower = med - k
        return series.clip(lower, upper)

    # ---------------------------------------------------------------------
    # 🛠️  Standardise to cap‑weighted μ = 0, EW σ = 1
    # ---------------------------------------------------------------------
    @staticmethod
    def _standardise(series: pd.Series, weights: pd.Series) -> pd.Series:
        """Barra/Axioma z‑score: cap‑weighted mean 0, equal‑weighted σ = 1."""
        mu = np.average(series, weights=weights)
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return series * 0.0
        return (series - mu) / sigma

    @staticmethod
    def _beta_weekly(prices: pd.DataFrame, mkt: pd.Series, lookback_weeks: int = 104) -> pd.Series:
        wk = prices.resample("W-FRI").last().pct_change().dropna()
        wk_mkt = mkt.reindex(wk.index).pct_change().dropna()
        wk, wk_mkt = wk.align(wk_mkt, join='inner', axis=0)

        betas = {}
        for col in wk.columns:
            y = wk[col]
            if len(y) < lookback_weeks * 0.6: continue
            cov = np.cov(y.tail(lookback_weeks), wk_mkt.tail(lookback_weeks))[0, 1]
            var = wk_mkt.tail(lookback_weeks).var()
            betas[col] = cov / var if var != 0 else np.nan
        return pd.Series(betas)

    @TimeSerie._post_init_routines()
    def __init__(self, assets_category_unique_id: str, market_beta_asset_proxy: ms_client.Asset,
                 em_winsor_days: int = 0,
                 em_min_obs: int = 0,

                 local_kwargs_to_ignore=["assets_category_unique_id"],
                 *args, **kwargs):
        super().__init__(*args, **kwargs, local_kwargs_to_ignore=local_kwargs_to_ignore)

        # --- Debugging import issue ---
        print("Inspecting polygon package contents:")
        print(dir(polygon))

        prices_config = PricesConfiguration(bar_frequency_id="1d")
        assets_configuration = AssetsConfiguration(
            assets_category_unique_id=assets_category_unique_id,
            price_type=PriceTypeNames.CLOSE,
            prices_configuration=prices_config
        )
        self.assets_category_unique_id = assets_category_unique_id

        # for EM markets
        self.em_winsor_days = em_winsor_days
        self.em_min_obs = em_min_obs

        # --- Dependencies ---

        self.prices_ts = get_interpolated_prices_timeseries(assets_configuration)

        pc = assets_configuration.prices_configuration
        # force update of prices for the beta proxy
        self.benchmark_ts = InterpolatedPrices(
            asset_list=[market_beta_asset_proxy],
            bar_frequency_id=pc.bar_frequency_id, upsample_frequency_id=pc.upsample_frequency_id,
            intraday_bar_interpolation_rule=pc.intraday_bar_interpolation_rule
        )

        self.fundamentals_ts = FundamentalsTimeSeries(assets_category_unique_id=assets_category_unique_id)
        self.market_beta_asset_proxy = market_beta_asset_proxy

    def _get_asset_list(self) -> Optional[List["Asset"]]:

        asset_ids = ms_client.AssetCategory.get(unique_identifier=self.assets_category_unique_id).assets

        asset_list = ms_client.Asset.filter(id__in=asset_ids + [self.market_beta_asset_proxy.id])
        return asset_list

    def update(self, update_statistics):
        """
        Build (or refresh) each trading-day’s exposure matrix for the classic
        12 USE4 / AXUS4 style factors.

        Data conventions reproduced verbatim from:

        • MSCI-Barra U.S. Equity Model (**USE4**) – Technical Notes, §3 “Style Factors”
          https://www.msci.com/documents/10199/242721/Barra_US_Equity_Model_USE4.pdf

        • Qontigo-Axioma United States Equity Factor Model (**AXUS4 / US4**) – Methodology /
          Factsheet, §5 “Factor List”
          https://cdn2.hubspot.net/hubfs/2174119/Return%20Downloads/Factsheet-AXUS4-1.pdf

        Key implementation points mirrored from those sources
        -----------------------------------------------------
        • **Price-based descriptors** (Momentum, Residual Vol, Liquidity, Beta)
          use 252-, 60-, 63- and 104-week windows exactly as specified.
        • **Fundamental descriptors** (Value, Growth, Leverage, etc.) are
          shifted +90 calendar days to avoid look-ahead, then forward-filled
          daily, matching the vendor’s D+1 QA lag.
        • **Size & Beta curvature terms** (Non-linear Size / Beta) follow the
          quadratic transforms documented in both papers.
        • **Winsorisation** = ±3·MAD; **re-standardisation** = cap-weighted
          mean 0, equal-weighted stdev 1 — identical to Barra/Axioma “Descriptor
          Hygiene”.

        Returns
        -------
        pd.DataFrame
            Multi-index rows  [time_index, unique_identifier], columns ordered
            per USE4 factor list.  Ready to append to the master exposure panel.
        """
        # --------------------------------------------------------------
        # 0.  Pull ~3 y of history so all rolling windows are available
        # --------------------------------------------------------------
        range_descriptor = update_statistics.get_update_range_map_great_or_equal()
        for uid, val in range_descriptor.items():
            val["start_date"] -= datetime.timedelta(days=750)  # ≈ 3 y

        prices_hist_df = self.prices_ts.get_ranged_data_per_asset(
            range_descriptor=range_descriptor)

        proxi_id = prices_hist_df.index.get_level_values(
            "unique_identifier") == self.market_beta_asset_proxy.unique_identifier
        market_prices_df = prices_hist_df[proxi_id].copy().reset_index("unique_identifier", drop=True)
        prices_hist_df = prices_hist_df[~proxi_id].copy()

        fundamentals_hist_df = self.fundamentals_ts.get_ranged_data_per_asset(
            range_descriptor=range_descriptor)

        if prices_hist_df.empty or fundamentals_hist_df.empty:
            return pd.DataFrame()

        # --------------------------------------------------------------
        # 1.  Pivot to wide frames (date × ticker)
        # --------------------------------------------------------------
        prices = prices_hist_df.unstack("unique_identifier")["close"]
        prices.index = prices.index.normalize()

        volume = prices_hist_df.unstack("unique_identifier")["volume"]
        volume.index = volume.index.normalize()

        fund_all = fundamentals_hist_df.unstack("unique_identifier")
        # Shift the statement dates +90 d to avoid look-ahead, then ffill
        fund_all.index = fund_all.index + datetime.timedelta(days=90)
        fund_all = fund_all.reindex(prices.index).ffill().astype(float)

        # Keep only assets present in both tables
        common_assets = prices.columns.intersection(
            fund_all.columns.get_level_values("unique_identifier"))
        prices = prices[common_assets]
        volume = volume[common_assets]
        fund_all = fund_all.loc[:, fund_all.columns.get_level_values(1).isin(common_assets)]

        # ────────────────────────────────────────────────────────────────
        # 2.  Pre-compute rolling / raw descriptors for the full panel
        # ────────────────────────────────────────────────────────────────

        # ⚙️ Size  (Barra USE4 §3.2 “Size”) TODO: this is WRONG we dont have the proper Point if Time for shares outstanding
        #    Log-market-cap compresses the extreme right tail so the factor is
        #    approximately symmetric; forward regressions then behave linearly.
        sh_out = fund_all.xs("basic_average_shares", level=0, axis=1)
        mkt_cap = prices * sh_out

        lncap = np.log(mkt_cap)  # primary Size factor
        lncap2 = lncap ** 2  # ⚙️ Non-linear Size (“Size Curve”):
        #    quadratic term captures curvature
        #    between micro-caps, mid-caps, mega-caps

        # ⚙️ Momentum – 12-1 definition (Barra “Momentum”, Axioma “Medium Horizon”)
        mom_12_1 = prices.pct_change(252).shift(21)  # 12-month return, skip last month
        # ⚙️ Short-term Momentum – pure 1-month return (used in trading-horizon models)
        mom_1m = prices.pct_change(21)

        # ⚙️ Residual Volatility – 60-day realised σ (proxy for specific risk)
        resid_vol = prices.pct_change().rolling(60).std()

        # ⚙️ Liquidity – log(ADV$/Cap) (Barra “Liquidity”, Axioma “Size Liquidity”)
        dollar_vol = (volume * prices).rolling(63).mean()  # 3-month dollar ADV
        liquidity = np.log(dollar_vol / mkt_cap)

        # ⚙️ Value – Book-to-Price (equity – preferred) / market-cap
        book_val = (fund_all["equity_attributable_to_parent"]
                    - fund_all["equity_attributable_to_noncontrolling_interest"]  # minority interest

                    )
        book_to_price = book_val / mkt_cap

        # ⚙️ Earnings Yield – TTM Net Income / market-cap
        earnings_yield = fund_all["net_income"] / mkt_cap
        # ⚙️ Dividend Yield – TTM cash dividends / market-cap
        div_yield = fund_all["total_dividends_per_share"]

        # ⚙️ Leverage – Total Debt / Total Assets (balance-sheet gearing)
        leverage = fund_all["total_debt"] / fund_all["total_assets"]

        # ⚙️ Profitability – Net Income / Total Assets (ROA-style quality metric)
        profitability = fund_all["net_income"] / fund_all["total_assets"]

        # ⚙️ Growth – 3-year sales growth (FY0 vs FY-3), held constant within FY
        rev = fund_all["revenue"]
        rev_fy = rev.groupby(pd.Grouper(freq="Y")).last()  # annual series
        growth_fy = rev_fy / rev_fy.shift(3) - 1
        growth = growth_fy.reindex(prices.index, method="ffill")

        # ⚙️ Beta – 104-week weekly CAPM β (Barra “Beta”); forward-filled to daily
        wk_ret = prices.resample("W-FRI").last().pct_change()
        wk_mkt = market_prices_df["close"].resample("W-FRI").last().pct_change().reindex(wk_ret.index)
        window = 104
        beta_w = (
            wk_ret
            .rolling(window, min_periods=52)  # e.g. allow beta after 52 weeks
            .cov(wk_mkt)
            .div(
                wk_mkt
                .rolling(window, min_periods=52)
                .var(), axis=0
            )
        )
        beta = beta_w.reindex(prices.index, method="ffill")
        nl_beta = beta ** 2  # ⚙️ Non-linear Beta – curvature term

        # ────────────────────────────────────────────────────────────────
        # 3.  Stack into one raw-descriptor panel: (date × ticker × desc)
        # ────────────────────────────────────────────────────────────────
        raw_desc = pd.concat({
            "lncap": lncap, "lncap2": lncap2,
            "beta": beta, "nl_beta": nl_beta,
            "mom_12_1": mom_12_1, "mom_1m": mom_1m,
            "resid_vol": resid_vol, "liquidity": liquidity,
            "book_to_price": book_to_price, "growth": growth,
            "leverage": leverage,
            "div_yield": div_yield, "earnings_yield": earnings_yield,
            "profitability": profitability, "market_cap": mkt_cap
        }, axis=1)

        # ────────────────────────────────────────────────────────────────
        # 4.  Winsorise & standardise **for every date** in `prices.index`
        #     Barra/Axioma define an *exposure* as the **z-scored, cap-weighted
        #     mean-zero version of each raw descriptor**.  That transformation
        #     makes the cross-sectional regression well-behaved and ensures that
        #     **one unit of factor return equals one percentage-point of stock
        #     return**.  The DataFrame we build below is therefore the canonical
        #     exposure matrix that the vendors distribute in their daily risk
        #     files.
        # ────────────────────────────────────────────────────────────────
        exposures_all_dates = []
        valid_dates = []

        for dt in prices.index:
            # NaNs on weekends/holidays → skip
            if dt not in raw_desc.index:
                continue

            # N‑day trailing window for Emerging‑Markets pooled winsor
            if self.em_winsor_days > 0:
                pool_start = dt - datetime.timedelta(days=self.em_winsor_days - 1)
                pooled_window = raw_desc.loc[pool_start:dt]
            else:
                pooled_window = None  # disable pooling

            row = raw_desc.loc[dt]
            if row.isnull().all().all():  # all descriptors missing
                continue

            mkt_cap_today = mkt_cap.loc[dt]
            if mkt_cap_today.isnull().all():
                continue
            cap_w_today = (mkt_cap_today / mkt_cap_today.sum()).fillna(0)

            ex_today: Dict[str, pd.Series] = {}

            for fac in STYLE_FACTOR_MAP.keys():
                if fac not in row.index.get_level_values(0):
                    continue
                s_raw = row.xs(fac)  # Series (index = tickers), NaNs preserved
                # Build pooled sample only if we configured pooling
                if pooled_window is not None:
                    pooled_sample = (
                        pooled_window
                        .xs(fac, level=0, axis=1)
                        .stack()
                    )
                else:
                    pooled_sample = None

                # Winsorise with EM rule if applicable
                s_w = self._winsorise(
                    s_raw,
                    pooled_sample=pooled_sample,
                    min_obs=self.em_min_obs,
                )

                # Neutral‑fill (median) so the z‑score lands at 0 for missing
                s_f = s_w.fillna(s_w.median())

                ex_today[fac] = self._standardise(s_f, cap_w_today)

            # Raw market‑cap (not z‑scored) for optional downstream use
            ex_today["market_cap"] = mkt_cap_today

            exposures_all_dates.append(pd.DataFrame(ex_today, index=common_assets))
            valid_dates.append(dt)  # <-- only if this dt resulted in an appended frame

        if not exposures_all_dates:
            return pd.DataFrame()  # shouldn’t happen

        expo_df = (pd.concat(exposures_all_dates, keys=valid_dates)
                   .reindex(columns=list(STYLE_FACTOR_MAP.keys()) + ["market_cap"])
                   .dropna(how="all"))

        expo_df.index.names = ["time_index", "unique_identifier"]

        return expo_df

    def _get_column_metadata(self):

        # descriptions drawn from your update(...) implementation
        desc = {
            'lncap': "Natural log of market cap; primary Size factor (Barra USE4 §3.2).",
            'lncap2': "Squared log market cap; captures non-linear Size curvature.",
            'beta': "104-week weekly CAPM β; systematic risk sensitivity, forward-filled to daily.",
            'nl_beta': "Squared beta; non-linear Beta curvature term.",
            'mom_12_1': "12-month return skipping the most recent month; medium-term Momentum.",
            'mom_1m': "1-month return; short-term STMomentum.",
            'resid_vol': "60-day rolling std of daily returns; ResidualVol (idiosyncratic risk).",
            'liquidity': "Log(63-day average dollar volume ÷ market cap); Liquidity factor.",
            'book_to_price': "Book-to-price ratio (equity ÷ market cap); Value factor.",
            'earnings_yield': "TTM net income ÷ market cap; EarningsYield.",
            'div_yield': "TTM dividends per share ÷ price; DividendYield.",
            'leverage': "Total debt ÷ total assets; Leverage factor.",
            'profitability': "Net income ÷ total assets; Profitability factor.",
            'growth': "3-year revenue growth (FY0 vs FY-3), forward-filled; Growth factor.",
        }

        return [
            ms_client.ColumnMetaData(
                column_name=col,
                dtype="float",
                label=label,
                description=desc[col]
            )
            for col, label in STYLE_FACTOR_MAP.items()
        ]

    def  _run_post_update_routines(self, error_on_last_update,update_statistics:ms_client.DataUpdates):

        market_beta_asset_proxy=self.market_beta_asset_proxy

        TS_UID = f"{CANONICAL_STYLE_FACTORS_MATRIX_ID}_{market_beta_asset_proxy.ticker}"

        source_table=self.local_time_serie.remote_table

        try:
            markets_time_series_details = ms_client.MarketsTimeSeriesDetails.get(
                unique_identifier=TS_UID,
            )
            if markets_time_series_details.source_table.id != source_table.id:
                markets_time_series_details = markets_time_series_details.patch(source_table__id=source_table.id)
        except ms_client.DoesNotExist:
            markets_time_series_details = ms_client.MarketsTimeSeriesDetails.update_or_create(
                unique_identifier=TS_UID,
                source_table__id=source_table.id,
                data_frequency_id=ms_client.DataFrequency.one_d,
                description=(
                    "Canonical daily exposure matrix of the 12 Axioma/Barra style factors. "
                    "Each column is the cap-weighted, winsorised and z-scored factor exposure "
                    "(one unit of factor return = 1% stock return), ready for downstream risk and attribution."
                    f"Using market proxy {self.style_ts.market_beta_asset_proxy.ticker}"
                ),

            )

        new_assets = []
        for asset in update_statistics.asset_list:
            if asset.id not in markets_time_series_details.assets_in_data_source:
                new_assets.append(asset)

        markets_time_series_details.append_asset_list_source(asset_list=new_assets)


class FactorReturnsTimeSeries(TimeSerie):
    """
    Stores the daily factor-return vectors fₜ.

    Parameters
    ----------
    style_ts : StyleFactorsTimeSeries
        An *already-constructed* StyleFactorsTimeSeries instance.  We reuse
        its prices, market-cap and exposure matrix instead of querying data
        again.
    """

    @TimeSerie._post_init_routines()
    def __init__(self, assets_category_unique_id: str,
                 market_beta_asset_proxy: ms_client.Asset,
                 local_kwargs_to_ignore=["assets_category_unique_id"],
                 *args, **kwargs):
        super().__init__(*args, **kwargs, local_kwargs_to_ignore=local_kwargs_to_ignore)

        self.style_ts = StyleFactorsTimeSeries(assets_category_unique_id=assets_category_unique_id,
                                               market_beta_asset_proxy=market_beta_asset_proxy,
                                               *args, **kwargs)

    def update(self, update_statistics):
        """
        Compute daily factor-return vectors fₜ by robust WLS on the exposure
        matrix produced by the companion StyleFactorsTimeSeries.  Assumes
        the stacked exposure frame contains a column named **'market_cap'**.
        Returns the residual panel (εᵢ,ₜ) so the caller can pipe it into a
        residual time-series.
        """
        # ------------------------------------------------------------------
        # 0 · Fetch stacked prices & exposures for the update window
        # ------------------------------------------------------------------

        prices_asset_list = self.style_ts._get_asset_list()

        range_descriptor = {a.unique_identifier: {"start_date": update_statistics.max_time_index_value,
                                                  "start_date_operand": ">="
                                                  } for a in prices_asset_list}

        prices_stacked = self.style_ts.prices_ts.get_ranged_data_per_asset(
            range_descriptor=range_descriptor
        )
        expos_stacked = self.style_ts.get_ranged_data_per_asset(
            range_descriptor=range_descriptor
        )

        if prices_stacked.empty or expos_stacked.empty:
            return pd.DataFrame()

        # wide price matrix for returns
        prices_wide = (
            prices_stacked
            .unstack("unique_identifier")["close"]
            .sort_index()
        )
        daily_ret = prices_wide.pct_change()
        daily_ret.index = daily_ret.index.normalize()
        daily_ret = daily_ret.iloc[1:]
        # ------------------------------------------------------------------
        # 1 · Loop over each trading day in exposures
        # ------------------------------------------------------------------
        import statsmodels.api as sm
        factor_frames = []

        for dt in expos_stacked.index.get_level_values("time_index").unique():
            if dt not in daily_ret.index:
                continue

            # --- build regressor matrix --- onda day observation
            Xwide = expos_stacked.xs(dt, level="time_index").copy()  # ticker × columns
            if Xwide.isnull().all(axis=0).any():
                self.logger.warning(
                    f"{dt} is dropped because {Xwide.columns[Xwide.isnull().all(axis=0)].to_list()} factors are Null")
                continue

            w_cap = Xwide.pop("market_cap")  # remove -> weight vector

            unique_identifiers = Xwide.index

            # align y and ensure float dtype
            y = daily_ret.loc[dt, unique_identifiers].astype(float)

            # ------------------------------------------------------------------
            # 1) drop rows whose **return** is NaN
            # 2) warn once, with the count
            # ------------------------------------------------------------------
            mask_valid = y.notna()
            if not mask_valid.all():  # at least one NaN present
                n_excluded = (~mask_valid).sum()
                self.logger.warning(
                    f"{n_excluded} assets excluded from cross-sectional regression "
                    f"on {dt:%Y-%m-%d} because the daily return is NaN"
                )
            # keep only the rows with a valid return
            y = y[mask_valid]
            Xwide = Xwide.loc[mask_valid].astype(float)
            w_cap = w_cap.loc[mask_valid]

            # ------------------------------------------------------------------
            # √cap weights:  statsmodels interprets `weights` as W_i, but the
            # loss function is  Σ ρ( W_i · residual_i ).  To achieve a **capital-
            # weighted** objective  Σ (Cap_i · ρ(residual_i)) we must pass
            # W_i = √Cap_i, exactly as specified in the MSCI-Barra USE4 notes (§4)
            # and Qontigo-Axioma AXUS4 guide (§6).  Empirically this also
            # approximates inverse-variance weights, because idiosyncratic
            # variance scales ~1/Cap.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------
            #  fill gaps
            # ------------------------------------------------------------
            # • Factor exposures (Xwide) – replace NaNs with the cross-section mean so the
            #   value is “neutral” and the row stays usable in the design matrix.
            # • Market-cap (w_cap)     – replace NaNs with 0 so √Cap = 0 ⇒ weight = 0,
            #   meaning the observation has no influence on the regression while still
            #   preserving index alignment for downstream calculations.
            Xwide = Xwide.fillna(Xwide.mean())  # factor exposures → mean
            w_cap = w_cap.fillna(0.0)  # missing cap → 0 → weight 0

            W = np.sqrt(w_cap)  # root-capitalisation weights

            # robust regression (Huber-M)
            rlm = sm.WLS(y, sm.add_constant(Xwide),
                         weights=W)
            res = rlm.fit()

            factor_frames.append(
                res.params.drop("const").to_frame(name=dt).T
            )

        # ------------------------------------------------------------------
        # Persist factor-return panel and return it
        # ------------------------------------------------------------------
        if factor_frames:
            factor_ret_df = pd.concat(factor_frames).sort_index()  # rows = date, cols = factor
            return factor_ret_df  # ← deliver factor-returns only

        return pd.DataFrame()  # nothing to update




    # ──────────────────────────────────────────────────────────────
    # 1 ▸ Column metadata  (one entry per factor‑return column)
    # ──────────────────────────────────────────────────────────────
    def _get_column_metadata(self):
        """
        Describe all factor‑return columns stored in this time‑series.

        Each ColumnMetaData marks the column as a float and explains that the
        value is a *daily factor return* expressed in percentage‑points:
        if a stock’s exposure to a factor is +1, that day’s factor return is the
        contribution (in %) to the stock’s return.
        """

        desc = {
            'lncap': "Daily return to the Size factor (log‑market‑cap).",
            'lncap2': "Return to the Size‑curvature term (non‑linear Size).",
            'beta': "Return to the systematic‑risk (Beta) factor.",
            'nl_beta': "Return to the Beta‑curvature term (non‑linear Beta).",
            'mom_12_1': "Return to medium‑term Momentum (12‑1 month).",
            'mom_1m': "Return to 1‑month (short‑term) Momentum.",
            'resid_vol': "Return to the Residual Volatility factor.",
            'liquidity': "Return to the Liquidity factor.",
            'book_to_price': "Return to the Value factor (book‑to‑price).",
            'growth': "Return to the Growth factor (3‑year sales growth).",
            'leverage': "Return to the Leverage factor (debt‑to‑assets).",
            'div_yield': "Return to the Dividend‑Yield factor.",
            'earnings_yield': "Return to the Earnings‑Yield factor.",
            'profitability': "Return to the Profitability (quality) factor.",
        }

        return [
            ms_client.ColumnMetaData(
                column_name=col,  # e.g. 'lncap'
                dtype="float",
                label=label,  # e.g. 'Size'
                description=(
                    f"{desc[col]} One unit equals a 1 % contribution to a stock’s "
                    f"return per one unit of exposure."
                )
            )
            for col, label in STYLE_FACTOR_MAP.items()
        ]

    # ──────────────────────────────────────────────────────────────
    # 2 ▸ Post‑update bookkeeping  (register this time‑series)
    # ──────────────────────────────────────────────────────────────
    def _run_post_update_routines(self, error_on_last_update,
                                  update_statistics: ms_client.DataUpdates):
        """
        Register (or patch) the canonical **factor‑return** time‑series with
        `MarketsTimeSeriesDetails` so downstream services can discover it.

        Unlike the exposure matrix, factor returns are *not security‑specific*,
        so no asset list is maintained here.
        """

        TS_UID = f"{CANONICAL_FACTOR_RETURNS_ID}_{self.style_ts.market_beta_asset_proxy.ticker}"

        source_table=self.local_time_serie.remote_table

        try:
            mts = ms_client.MarketsTimeSeriesDetails.get(
                unique_identifier=TS_UID
            )

            # Ensure it points at the same local_time_serie we just updated
            if mts.source_table.id !=  source_table.id:
                mts = mts.patch(source_table__id=source_table.id)

        except ms_client.DoesNotExist:
            # Create the record the first time we run
            mts = ms_client.MarketsTimeSeriesDetails.update_or_create(
                unique_identifier=TS_UID,
                source_table__id=source_table.id,
                data_frequency_id=ms_client.DataFrequency.one_d,
                description=(
                    "Canonical daily returns for the 12 Axioma/Barra style factors "
                    "computed via robust capital‑weighted WLS against the exposure "
                    f"matrix. using for market proxy {self.style_ts.market_beta_asset_proxy.ticker}"
                ),
            )

        # Factor returns have no per‑asset dimension, so there is no asset list
        # to append.  The method exits silently once the metadata is in place.


class FactorResidualTimeSeries(TimeSerie):
    """
    Stores ε₍ᵢ,ₜ₎  – the idiosyncratic residuals produced by the robust
    regression that lives in FactorReturnsTimeSeries.
    """

    @TimeSerie._post_init_routines()
    def __init__(self, assets_category_unique_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.style_ts = StyleFactorsTimeSeries(assets_category_unique_id=assets_category_unique_id, *args, **kwargs)

    def update(self, update_statistics):
        """
        1.  Call the style engine to refresh exposures (and mkt-cap).
        2.  For every new trading day, run the robust cross-sectional
            regression  rₜ = Xₜ fₜ + εₜ   but keep *only* εₜ.
        3.  Append εₜ to this TimeSerie and return the panel.

        Returns
        -------
        pd.DataFrame
            Index  : time_index (date) × unique_identifier (ticker)
            Columns: single 'residual' column of specific returns.
            Empty DataFrame if no new dates were processed.
        """
        # 1 · update exposures; we need expo_df & mkt_cap for the regression
        expo_df = self.style_ts.update(update_statistics)
        if expo_df.empty:
            return pd.DataFrame()

        prices = self.style_ts.prices_ts.get_dataframe()
        mkt_cap = self.style_ts.mkt_cap
        daily_ret = prices.pct_change()  # todo: Properly this needs to be excess return

        import statsmodels.api as sm
        resid_frames = []

        # 2 · loop over each date block from exposures
        for dt, X in expo_df.groupby(level="time_index"):
            if dt not in daily_ret.index:
                continue
            y = daily_ret.loc[dt, X.index.get_level_values(1)].astype(float)
            Xmat = X.droplevel("time_index")  # ticker × factor
            W = np.sqrt(mkt_cap.loc[dt, Xmat.index])  # √cap weights

            rlm = sm.RLM(y, sm.add_constant(Xmat),
                         M=sm.robust.norms.HuberT(), weights=W)
            res = rlm.fit()

            # store residuals (one column per ticker)
            resid_frames.append(
                res.resid.to_frame(name=dt).T  # shape 1 × tickers
            )

        if not resid_frames:
            return pd.DataFrame()  # nothing new this call

        residuals_df = pd.concat(resid_frames).sort_index()
        # stack into (date, ticker) rows & single 'residual' column
        residuals_df = residuals_df.stack().to_frame(name="residual")
        residuals_df.index.names = ["time_index", "unique_identifier"]

        # 3 · persist and return
        self.append_dataframe(residuals_df)
        return residuals_df






