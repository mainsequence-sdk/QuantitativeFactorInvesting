



class PortfolioFactorAnalysis:
    """
    Perform factor-based analyses on a portfolio given:

      - factor_returns_ts: a FactorReturnsTimeSeries instance that:
          * Provides historical factor returns via get_df_between_dates()
          * Exposes style_ts for factor exposures and prices via .style_ts

      - portfolio_weights: a pandas Series mapping each asset's unique_identifier to its weight
          * If None, a random portfolio is constructed from the 'magnificent_7' AssetCategory

      - start_date, end_date: datetime bounds for fetching all historical data

    Attributes
    ----------
    exposures_df : pd.DataFrame
        MultiIndex (time_index, unique_identifier) × factor columns, holds per-asset factor exposures

    factor_returns_df : pd.DataFrame
        Index = time_index, columns = factor names, holds realized factor returns

    asset_returns_df : pd.Series
        MultiIndex (time_index, unique_identifier) → asset-level returns, computed from prices

    portfolio_exposure_df : pd.DataFrame
        Index = time_index, columns = factor names, holds portfolio-level exposure per factor
    """

    def __init__(self,
                 factor_returns_ts: FactorReturnsTimeSeries,
                 portfolio_weights: pd.Series = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None):
        """
        Initialize the analysis with data sources and optional portfolio weights.

        Parameters
        ----------
        factor_returns_ts : FactorReturnsTimeSeries
            Supplies factor returns and has .style_ts for exposures & prices.
        portfolio_weights : pd.Series, optional
            Weights of assets in the portfolio (indexed by unique_identifier). Must sum to 1.
            If None, a random portfolio from 'magnificent_7' is generated.
        start_date : datetime.datetime
            Start date (inclusive) for data retrieval.
        end_date : datetime.datetime
            End date (inclusive) for data retrieval.
        """
        # --- Data sources setup ---
        self.factor_returns_ts = factor_returns_ts
        self.exposures_ts      = factor_returns_ts.style_ts
        self.prices_ts         = factor_returns_ts.style_ts.prices_ts

        # --- Portfolio weights defaulting ---
        if portfolio_weights is None:
            portfolio_weights = self._create_random_portfolio()
        self.portfolio_weights = portfolio_weights

        # --- Date range for all historical queries ---
        self.start_date = start_date
        self.end_date   = end_date

        # ---------- lazy-loaded caches ----------
        self._exposures_df: pd.DataFrame | None = None
        self._factor_ret_df: pd.DataFrame | None = None
        self._asset_ret_series: pd.Series | None = None
        self._port_expo_df: pd.DataFrame | None = None  # portfolio exposures

    # ------------------------------------------------------------------
    # 2)  LAZY-LOADED PROPERTIES
    # ------------------------------------------------------------------
    @property
    def exposures_df(self) -> pd.DataFrame:
        """
        Asset × factor exposures over the chosen date range.

        Cached after first download so repeated access is free.
        """
        if self._exposures_df is None:
            df = self.exposures_ts.get_df_between_dates(
                start_date=self.start_date,
                end_date=self.end_date
            ).drop(columns=['market_cap'], errors='ignore')
            self._exposures_df = df
        return self._exposures_df

    @property
    def factor_returns_df(self) -> pd.DataFrame:
        """
        Daily factor returns over the chosen date range (cached).
        """
        if self._factor_ret_df is None:
            fr = self.factor_returns_ts.get_df_between_dates(
                start_date=self.start_date,
                end_date=self.end_date
            )
            self._factor_ret_df = fr
        return self._factor_ret_df

    @property
    def asset_returns_df(self) -> pd.Series:
        """
        Daily asset returns (cached).  Constructed from the prices_ts source.
        """
        if self._asset_ret_series is None:
            prices = (
                self.prices_ts
                .get_df_between_dates(self.start_date, self.end_date)[['close']]
            )
            times   = pd.to_datetime(prices.index.get_level_values('time_index')).normalize()
            assets  = prices.index.get_level_values('unique_identifier')
            prices.index = pd.MultiIndex.from_arrays([times, assets],
                                                     names=['time_index', 'unique_identifier'])
            self._asset_ret_series = (
                prices['close']
                .groupby(level='unique_identifier')
                .pct_change()
                .dropna()
                .rename('return')
            )
        return self._asset_ret_series



    @property
    def portfolio_exposure_df(self) -> pd.DataFrame:
        """
        Portfolio-level factor exposures, cached after first computation.
        """
        if self._port_expo_df is None:
            df = (
                self.exposures_df
                .reset_index()
                .merge(self.portfolio_weights.rename('weight'),
                       left_on='unique_identifier', right_index=True)
                .set_index(['time_index', 'unique_identifier'])
            )
            factors   = [c for c in df.columns if c != 'weight']
            wtd       = df[factors].multiply(df['weight'], axis=0)
            self._port_expo_df = wtd.groupby(level='time_index').sum()
        return self._port_expo_df

    def _create_random_portfolio(self) -> pd.Series:
        """
        Construct random weights for assets in the fixed 'magnificent_7' category.

        Returns
        -------
        pd.Series
            Random weights (sum to 1), indexed by each asset's unique_identifier.
        """
        cat = ms_client.AssetCategory.get(unique_identifier="magnificent_7")
        assets = ms_client.Asset.filter(id__in=cat.assets)
        ids = [a.unique_identifier for a in assets]
        w = np.random.rand(len(ids))
        w /= w.sum()
        return pd.Series(w, index=ids, name='weight')






    def rolling_statistics(self, window: int = 60):
        """
        Compute rolling summary metrics on portfolio factor exposures.

        Intention
        ---------
        Use a sliding window of `window` periods to measure:
          - How average exposure changes over the window (rolling mean)
          - How stable exposures are (rolling standard deviation)
          - How persistent the exposures are day-to-day (rolling 1-day autocorrelation)

        Parameters
        ----------
        window : int, default=60
            The lookback length for computing the rolling window.

        Returns
        -------
        rolling_mean : pd.DataFrame
            The mean exposure for each factor over each rolling window.
        rolling_std : pd.DataFrame
            The standard deviation of each factor's exposures over each window.
        rolling_acf : pd.DataFrame
            The lag-1 autocorrelation of each factor's exposures over each window.
        """
        roll = self.portfolio_exposure_df.rolling(window=window)
        rm = roll.mean()
        rv = roll.std()
        # The following measures how correlated today's exposures are with yesterday's
        racf = self.portfolio_exposure_df.rolling(window).apply(lambda x: x.autocorr(lag=1))
        return rm, rv, racf

    def factor_attribution(self):
        """
        Perform ex-post factor P&L attribution.

        Intention
        ---------
        Join portfolio-level exposures with realized factor returns, then calculate
        each factor's P&L contribution by multiplying exposures by returns, and
        aggregate to get the model-predicted portfolio return.

        Returns
        -------
        contrib_df : pd.DataFrame
            Date×factor matrix of per-factor contributions.
        predicted_returns : pd.Series
            Total factor-model return per date.
        """
        df = self.portfolio_exposure_df.join(
            self.factor_returns_df,
            how='inner',
            rsuffix='_ret'
        )
        ret_cols = [c for c in df if c.endswith('_ret')]
        factor_cols = [c for c in df if not c.endswith('_ret')]

        # Rename return columns to match factor names
        rets = df[ret_cols].rename(columns={c: c.replace('_ret','') for c in ret_cols})
        expo = df[factor_cols]

        contrib_df = expo * rets
        predicted_returns = contrib_df.sum(axis=1)
        return contrib_df, predicted_returns

    def exposure_tail_metrics(self,
                              date: pd.Timestamp,
                              tail_cut: float = 2.0) -> pd.DataFrame:
        """
        Gauge how 'fat' the exposure tails are on a given date — something that
        still matters even after z-scoring each factor cross-sectionally.

        For every factor f on date t we compute:

        *  tail_pos  = % of assets with  X_{i,f}(t)  >  tail_cut
        *  tail_neg  = % of assets with  X_{i,f}(t)  < −tail_cut
        *  kurtosis  = sample excess kurtosis of X_{i,f}(t)
        *  skewness  = sample skewness of X_{i,f}(t)

        Parameters
        ----------
        date : pd.Timestamp
            Snapshot date.
        tail_cut : float, default 2.0
            Z-score threshold that defines the “extreme” tail.

        Returns
        -------
        pd.DataFrame
            Index   = factor names
            Columns = ['tail_pos', 'tail_neg', 'kurtosis', 'skewness']
        """
        # ---- slice out the exposure matrix for that date ----
        X = self.exposures_df.xs(date, level='time_index')

        # ---- compute metrics ----
        tail_pos = (X > tail_cut).sum(axis=0) / len(X)
        tail_neg = (X < -tail_cut).sum(axis=0) / len(X)
        kurt = X.kurtosis(axis=0)  # excess kurtosis
        skew = X.skew(axis=0)

        return pd.concat(
            [tail_pos.rename('tail_pos'),
             tail_neg.rename('tail_neg'),
             kurt.rename('kurtosis'),
             skew.rename('skewness')],
            axis=1
        )

    def correlation_matrix(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute the cross-asset correlation matrix of factor exposures on a given date.

        Parameters
        ----------
        date : pd.Timestamp
            Target date for correlation calculation.

        Returns
        -------
        pd.DataFrame
            Asset×asset correlation of exposure profiles.
        """
        return self.exposures_df.xs(date, level='time_index').corr()

    def scenario_analysis(self, shock_vector: pd.Series, date: pd.Timestamp) -> pd.Series:
        """
        Estimate P&L impact under a hypothetical factor shock scenario.

        Intention
        ---------
        Multiply the portfolio's factor exposures by a user-specified shock vector
        to see how much each factor move would contribute to P&L on that date.

        Parameters
        ----------
        shock_vector : pd.Series
            Hypothetical factor returns (index=factor names).
        date : pd.Timestamp
            Date at which to apply the shock.

        Returns
        -------
        pd.Series
            P&L impact per factor = exposure × shock.
        """
        expo = self.portfolio_exposure_df.loc[date]
        return expo * shock_vector

    def estimate_realized_factor_premia(self,
                                        date: pd.Timestamp,
                                        asset_weights: pd.Series = None):
        """
        Estimate the realized cross‐sectional factor premia (and alpha) on a given date.

        This routine solves the cross‐sectional model:
            r_i(t) = α(t) + X_i(t)·β(t) + ε_i(t)
        where:
          • r_i(t) are asset returns at date t
          • X_i(t) are asset exposures to each factor at date t
          • β(t) are the implied factor premia (risk premia) at date t
          • α(t) is the cross‐sectional intercept (alpha)
          • ε_i(t) are residuals

        Features
        --------
        • Adds an intercept term to capture α(t).
        • Optionally uses WLS with `asset_weights` to downweight small or illiquid names.
        • Computes heteroskedasticity-consistent (HC1) standard errors.
        • Returns the condition number to flag collinearity among factors.

        Parameters
        ----------
        date : pd.Timestamp
            The snapshot date for the regression.
        asset_weights : pd.Series, optional
            Weights per asset (indexed by unique_identifier) for WLS.
            If None, performs ordinary OLS.

        Returns
        -------
        dict
            {
              "premia": pd.Series of β(t) and α(t),
              "stderr": pd.Series of HC1 robust standard errors,
              "tstat": pd.Series of t-statistics,
              "pval": pd.Series of p-values,
              "rsq": float R-squared of fit,
              "residuals": pd.Series of ε_i(t),
              "cond_num": float condition number of X'X
            }
        """
        # 1) snapshot exposures & returns
        X = self.exposures_df.xs(date, level='time_index').copy()
        y = self.asset_returns_df.xs(date, level='time_index')

        # 2) add intercept
        X.insert(0, 'alpha', 1.0)

        # 3) fit model (WLS or OLS) with HC1 robust errors
        if asset_weights is not None:
            w = asset_weights.reindex(X.index).fillna(0.0)
            model = sm.WLS(y, X, weights=w).fit(cov_type='HC1')
        else:
            model = sm.OLS(y, X).fit(cov_type='HC1')

        # 4) return structured results
        return {
            "premia": model.params,
            "stderr": model.bse,
            "tstat": model.tvalues,
            "pval": model.pvalues,
            "rsq": model.rsquared,
            "residuals": model.resid,
            "cond_num": model.condition_number
        }

    def compute_factor_covariance(self,
                                  method: str = 'ewma',
                                  window: int = 252,
                                  halflife: float = 63,
                                  shrinkage: float = None):
        """
        Compute the K×K factor covariance matrix at each date.

        Parameters
        ----------
        method : {'sample','ewma'}
          'sample'  = simple rolling-window sample covariance
          'ewma'    = exponentially-weighted covariance
        window : int
          Lookback length for rolling-sample method.
        halflife : float
          Half-life for EWMA weights (only if method='ewma').
        shrinkage : float or None
          If provided, shrink toward constant-correlation target by this intensity.

        Returns
        -------
        cov_ts : pd.DataFrame
          MultiIndex (time_index, factor_i, factor_j) → covariance values.
        """
        # 1) grab the factor returns series
        r = self.factor_returns_df
        r = r - r.mean()

        # 2) Compute base covariance panel: index=(date, factor_i), columns=factor_j
        if method == 'sample':
            cov_ts = r.rolling(window).cov()
        elif method == 'ewma':
            cov_ts = r.ewm(halflife=halflife, adjust=False).cov()
        else:
            raise ValueError("method must be 'sample' or 'ewma'")

        # 3) Extract the covariance matrix for the latest date
        last_date = cov_ts.index.get_level_values(0).max()
        Sigma = cov_ts.loc[last_date]  # DataFrame: factor_i × factor_j

        # 4) Apply shrinkage toward constant-correlation target if requested
        if shrinkage is not None:
            # Diagonal standard deviations
            stds = np.sqrt(np.diag(Sigma))
            # Build the constant-correlation target matrix
            corr_mat = Sigma.values / np.outer(stds, stds)
            avg_corr = corr_mat[np.triu_indices_from(corr_mat, k=1)].mean()
            target = pd.DataFrame(
                np.outer(stds, stds) * avg_corr,
                index=Sigma.index,
                columns=Sigma.columns
            )
            # Shrink
            Sigma = shrinkage * Sigma + (1 - shrinkage) * target

        return Sigma