
import mainsequence.client as ms_client
from factorinvesting.time_series.factors_time_series import *
from factorinvesting.src.analysis import PortfolioFactorAnalysis
import datetime

if __name__ == "__main__":
    market_asset = ms_client.Asset.get(ticker="IVV",
                                       execution_venue__symbol=ms_client.MARKETS_CONSTANTS.MAIN_SEQUENCE_EV,
                                       security_type=ms_client.MARKETS_CONSTANTS.FIGI_SECURITY_TYPE_ETP,
                                       security_market_sector=ms_client.MARKETS_CONSTANTS.FIGI_MARKET_SECTOR_EQUITY,
                                       )

    ts = FundamentalsTimeSeries(assets_category_unique_id='s&p500_constitutents')
    # ts.run(debug_mode=True, update_tree=False,force_update=True)

    style_ts = StyleFactorsExposureTS(assets_category_unique_id='s&p500_constitutents',
                                      market_beta_asset_proxy=market_asset,
                                      )
    # style_ts.run(debug_mode=True,update_tree=False)

    factor_returns_ts = FactorReturnsTimeSeries(assets_category_unique_id='s&p500_constitutents',
                                                market_beta_asset_proxy=market_asset,
                                                )
    factor_returns_ts.run(debug_mode=True, update_tree=True, force_update=True)
    # Specify analysis date range
    start = datetime.datetime(2022, 1, 1)
    end = datetime.datetime(2025, 1, 30)

    # Instantiate and run all analyses
    pfa = PortfolioFactorAnalysis(
        factor_returns_ts=factor_returns_ts,
        portfolio_weights=None,
        start_date=start,
        end_date=end
    )

