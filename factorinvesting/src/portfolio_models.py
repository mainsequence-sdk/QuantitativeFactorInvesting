import numpy as np
import pandas as pd
from sklearn import set_config
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior
from skfolio.moments import BaseMu, BaseCovariance
from skfolio.population import Population

import plotly.express as px


class FixedMu(BaseMu):
    """Estimator that returns a fixed expected returns vector."""
    def __init__(self, mu: pd.Series):
        super().__init__()
        self._mu = mu.copy()
    def __sklearn_clone__(self):
        # bypass cloning to preserve fixed mu
        return self
    def fit(self, X=None, y=None):
        self.mu_ = self._mu
        return self
    def get_params(self, deep=True):
        return {'mu': self._mu}
    def set_params(self, **params):
        if 'mu' in params:
            self._mu = params['mu']
        return self

class FixedCovariance(BaseCovariance):
    """Estimator that returns a fixed covariance matrix."""
    def __init__(self, cov: pd.DataFrame):
        super().__init__()
        self._cov = cov.copy()
    def __sklearn_clone__(self):
        # bypass cloning to preserve fixed covariance
        return self
    def fit(self, X=None, y=None):
        self.covariance_ = self._cov
        return self
    def get_params(self, deep=True):
        return {'cov': self._cov}
    def set_params(self, **params):
        if 'cov' in params:
            self._cov = params['cov']
        return self

class ExpandedPortfolioOptimizer:
    """
    ExpandedPortfolioOptimizer provides methods for constructing various optimal portfolios.

    Attributes
    ----------
    original_weights : pd.Series
        The weights computed by the last optimization method called.
    """

    def __init__(self,original_weights:pd.Series,profitability_score:pd.Series):
        self.original_weights = original_weights
        self.profitability_score=profitability_score

    @staticmethod
    def generate_random_data(
        asset_ids: list,
        seed: int = None
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Generate random expected returns and a positive-definite covariance matrix for given assets.

        Parameters
        ----------
        asset_ids : list of str
            List of asset identifiers.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        expected_returns : pd.Series
            Randomly generated expected returns (index: asset_ids).
        cov_matrix : pd.DataFrame
            Randomly generated positive-definite covariance matrix (index and columns: asset_ids).
        """
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        n = len(asset_ids)
        # Random expected returns, e.g., normal around 0.1 with std 0.2
        returns = np.random.normal(loc=0.1, scale=0.2, size=n)

        # Create random positive-definite covariance matrix:
        # Draw a random matrix and multiply by its transpose
        A = np.random.randn(n, n)
        cov = np.dot(A, A.T)

        # Optionally, scale variance to a reasonable level (e.g., variances ~0.05)
        # Here we normalize the diagonal to uniform variances
        stds = np.sqrt(np.diag(cov))
        cov = cov / np.outer(stds, stds) * 0.05  # set variances to 0.05

        # Build pandas structures
        expected_returns = pd.Series(returns, index=asset_ids)
        cov_matrix = pd.DataFrame(cov, index=asset_ids, columns=asset_ids)

        return expected_returns, cov_matrix
    def optimize_unconstrained(
            self,
            expected_returns: pd.Series,
            cov_matrix: pd.DataFrame,
            portfolio_name:str
    ) -> MeanRisk:
        """
        Fit a skfolio MeanRisk model for the global minimum-variance portfolio using only expected returns and covariance.

        Parameters
        ----------
        expected_returns : pd.Series
            Estimated expected returns (index: assets).
        cov_matrix : pd.DataFrame
            Estimated covariance matrix (assets x assets).

        Returns
        -------
        model : MeanRisk
            The fitted MeanRisk optimizer.
        """
        # enable metadata routing
        set_config(enable_metadata_routing=True)

        # align assets
        assets = expected_returns.index.intersection(cov_matrix.index)
        mu = expected_returns.loc[assets]
        cov = cov_matrix.loc[assets, assets]


        # wrap in metadata-routing estimators
        cov_est =FixedCovariance(cov).fit(None)
        prior = EmpiricalPrior( covariance_estimator=cov_est)

        # instantiate optimizer for min-vol
        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            prior_estimator=prior,
            threshold_long=0.15,
            solver="SCIP",portfolio_params=dict(name=portfolio_name)

        )

        # fit with empty DataFrame; moments routed via metadata
        X_dummy = mu.to_frame().T
        model.fit(X_dummy,)

        return model



    @staticmethod
    def compare_weights(
            model: MeanRisk,
            initial_weights: pd.Series
    ) -> "plotly.graph_objs._figure.Figure":
        """
        Compare initial and optimized weights, plotting only assets where
        the weight actually changed, ordered by magnitude of change, using Plotly.
        """
        # ensure feature names are present
        if not hasattr(model, 'feature_names_in_'):
            raise AttributeError(
                "Model does not have `feature_names_in_`."
                " Ensure you called optimize_unconstrained with a DataFrame so that feature names are set."
            )
        idx = list(model.feature_names_in_)
        new_weights = pd.Series(model.weights_, index=idx)

        # align initial and compute change
        aligned_initial = initial_weights.reindex(idx).fillna(0)
        delta = new_weights - aligned_initial
        changed = delta[delta != 0]

        if changed.empty:
            raise ValueError("No weight changes detected between initial and optimized portfolio.")

        # order assets by absolute change descending
        sorted_assets = changed.abs().sort_values(ascending=False).index.tolist()

        # prepare data frame
        df = pd.DataFrame({
            "initial": aligned_initial.loc[sorted_assets],
            "optimized": new_weights.loc[sorted_assets],
        }).reset_index().melt(
            id_vars="index",
            var_name="portfolio",
            value_name="weight"
        )
        df.rename(columns={"index": "asset"}, inplace=True)

        # enforce ordering on the x-axis
        df['asset'] = pd.Categorical(df['asset'], categories=sorted_assets, ordered=True)

        # plot
        fig = px.bar(
            df,
            x="asset",
            y="weight",
            color="portfolio",
            barmode="group",
            title="Portfolio Weights: Initial vs Optimized (Ordered by Change)"
        )
        fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Weight",
            legend_title_text="Portfolio"
        )

        return fig
    @staticmethod
    def plot_portfolios(portfolios:list) -> "plotly.graph_objs._figure.Figure":
        """
        Build a skfolio Population by supplying asset returns and each model's weights,
        then plot the efficient frontier.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns time series (index: timestamps; columns: asset ids).
        models : dict
            Mapping from model name to fitted MeanRisk instance.

        Returns
        -------
        fig : plotly Figure
            Population efficient frontier plot based on the provided returns and weights.
        """
        # initialize population
        population = Population(portfolios)
        fig = population.plot_cumulative_returns()

        return fig

