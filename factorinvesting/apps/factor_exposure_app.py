from mainsequence.virtualfundbuilder.resource_factory.app_factory import register_app, BaseApp, HtmlApp
from pydantic import BaseModel, Field
from typing import List, Optional
from factorinvesting.src.analysis import PortfolioFactorAnalysis
from factorinvesting.data_nodes.factors_time_series import *
import datetime
import mainsequence.client as ms_client
from mainsequence.reportbuilder.slide_templates import plot_dataframe_line_chart,generic_plotly_bar_chart
from mainsequence.client.models_tdag import Artifact
import json
import plotly.express as px
import tempfile




class ShockAnalysisConfiguration(BaseModel):
    factor_name: str #= Field(  title="Factor Name",  description="Factor column name ", example="beta",  )
    factor_shock_multiplier: float #= Field(   title="Factor Shock",
                                    #          description="The shock multiplier to be applied to a factor f*(1+factor_shock_multiplier)",
        #example=.05,
    #)


class FactorAnalysisConfiguration(BaseModel):
    """Pydantic model defining the parameters for report generation."""
    portfolio_ticker: str #= Field(
    #     title="Portfolio Ticker",
    #     description="Ticker of the target_portfolio reference ms_client.Portfolio",
    #     example="portfo446B",
    # ),
    folder_name: str #= Field(default="Factor Analysis Reports",
                             # title="Folder Name",
                             # description="Name of the folder where the report will be saved",
                             # example="Temp Folder",
                             # )

    calculate_historical_exposures: bool = True
    calculate_factor_attribution: bool = True
    calculate_tail_exposure: bool = True
    calculate_exposure_correlation_matrix: bool = True
    shocks_configuration: Optional[List[ShockAnalysisConfiguration]] = None
    start_date: str #= Field(
    #     ...,
    #     title="Start Date",
    #     description="Inclusive start datetime for the analysis (must include timezone)",
    #     example="2025-07-01T00:00:00+02:00",
    # )
    end_date: str #= Field(
    #     ...,
    #     title="End Date",
    #     description="Inclusive end datetime for the analysis (must include timezone)",
    #     example="2025-07-31T23:59:59+02:00",
    # )

    presentation_theme: Optional[str] #= Field(default="Main Sequence",
                                    # title="Presentation Theme",
                                    # description="Theme to use in the presentation",
                                    # example="Main Sequence",
                                    # )


@register_app()
class FactorExposureApp(HtmlApp):
    """
    Minimal example of a 'ReportApp' that can:
    1) Generate dummy data and create charts (line + heatmap).
    2) Embed those charts into an HTML template.
    3) Optionally export the HTML to PDF using WeasyPrint.
    """
    configuration_class = FactorAnalysisConfiguration

    def __init__(self, configuration: FactorAnalysisConfiguration):
        self.configuration = configuration

        # create folder if not exist
        self.folder = ms_client.Folder.get_or_create(name=self.configuration.folder_name)
        self.configuration_hash = self.hash_pydantic_object(self.configuration)
        super().__init__()
    def _chart_to_artifact(self, chart_html: str, suffix: str) -> str:
        """
        Write a Plotly-HTML chart to a temp file, upload it to Artifact storage,
        remove the temp file, and return the artifact id.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            tmp.write(chart_html)
            tmp.flush()
            temp_path = tmp.name

        try:
            artifact = Artifact.upload_file(
                filepath=temp_path,
                name=f"{suffix}_{self.configuration_hash}",
                created_by_resource_name=self.__class__.__name__,
                bucket_name="FactorExposureAssets",
            )
            return artifact.id
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    def _get_slide(self, presentation, index: int):
        """Return the slide at *index*, creating empty slides until it exists."""
        presentation=ms_client.Presentation.get(pk=presentation.id)
        if len(presentation.slides) <= index:
            slide=presentation.add_slide()
        else:
            slide=presentation.slides[index]
        return slide

    def _patch_slide(
            self,
            slide,
            title: str,
            artifact_id: str,
            note: str,
    ):
        """Build the Rich-Text body once and patch the slide."""
        body = {
            "type": "doc",
            "content": [
                ms_client.TextParagraph.heading(title, level=2, text_align="left").model_dump(),
                json.loads(ms_client.AppNode.make_app_node(id=artifact_id).model_dump_json()),
                ms_client.TextParagraph.paragraph(note, text_align="left").model_dump(),
            ],
        }
        slide.patch(body=json.dumps(body))

    def run(self):

        portfolio = ms_client.Portfolio.get(portfolio_ticker=self.configuration.portfolio_ticker)

        latest_weights = portfolio.get_latest_weights()

        latest_weights = pd.Series(latest_weights)
        market_asset = ms_client.Asset.get(ticker="IVV",
                                           execution_venue__symbol=ms_client.MARKETS_CONSTANTS.MAIN_SEQUENCE_EV,
                                           security_type=ms_client.MARKETS_CONSTANTS.FIGI_SECURITY_TYPE_ETP,
                                           security_market_sector=ms_client.MARKETS_CONSTANTS.FIGI_MARKET_SECTOR_EQUITY,
                                           )

        factor_returns_ts = FactorReturnsDataNodes(assets_category_unique_id='s&p500_constitutents',
                                                    market_beta_asset_proxy=market_asset,
                                                    )

        pfa = PortfolioFactorAnalysis(portfolio_weights=latest_weights,
                                      factor_returns_ts=factor_returns_ts,
                                      start_date=pd.to_datetime(self.configuration.start_date).replace(tzinfo=pytz.utc),
                                      end_date=pd.to_datetime(self.configuration.end_date).replace(tzinfo=pytz.utc)
                                      )

        #### report data

        presentation = ms_client.Presentation.get_or_create_by_title(
            title=f"{self.configuration.portfolio_ticker} Factor Analysis: {self.configuration.start_date} - {self.configuration.end_date}",
            folder=self.folder.id,
            description=f"Automatically generated presentation for for {self.configuration.portfolio_ticker}",
            )
        theme=ms_client.Theme.get(name=self.configuration.presentation_theme)
        presentation.patch(theme_id=theme.id)
        presentation.theme.set_plotly_theme()


        contrib, pred_returns = pfa.factor_attribution()
        last_date = contrib.index.max()
        # 1) Historical exposures ------------------------------------------------
        if self.configuration.calculate_historical_exposures:
            port_expo = pfa.portfolio_exposure_df.reset_index().melt(
                id_vars='time_index', var_name='Factor', value_name='Exposure'
            )
            fig = px.line(port_expo, x='time_index', y='Exposure', color='Factor',
                          title='Portfolio Factor Exposures Over Time')
            fig.update_layout(
                margin=dict(b=100),  # just enough room at the bottom
                legend_orientation='h',  # horizontal entries
                legend_x=0,  # align left
                legend_y=-0.2  # push below the plotting area
            )
            fig.update_traces(line_width=1)

            html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                               config={"responsive": True, "displayModeBar": False})
            art_id = self._chart_to_artifact(html, "historical_exposure")
            slide = self._get_slide(presentation, 0)
            self._patch_slide(
                slide,
                title="Portfolio Exposure",
                artifact_id=art_id,
                note=(
                    f"Exposures calculated from "
                    f"{self.configuration.start_date} – "
                    f"{self.configuration.end_date}"
                )
            )

        # 2) Factor attribution --------------------------------------------------
        if self.configuration.calculate_factor_attribution:
            bar = contrib.loc[last_date].reset_index(name="Contribution")
            fig = px.bar(bar, x='index', y='Contribution',
                         color="Contribution",  # <-- map bar color to the numeric value

                         title=f'Factor P&L Contributions on {last_date.date()}')
            html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                               config={"responsive": True, "displayModeBar": False})
            art_id = self._chart_to_artifact(html, "factor_contribution")
            slide = self._get_slide(presentation, 1)
            self._patch_slide(
                slide,
                title="Factor Contribution",
                artifact_id=art_id,
                note=f"Exposure at date {last_date}",
            )

        # 3) Tail-exposure heat map ---------------------------------------------
        if self.configuration.calculate_tail_exposure:
            tail_metrics = pfa.exposure_tail_metrics(last_date, tail_cut=2.0)
            plot_df = tail_metrics.reset_index().melt(
                id_vars="index",
                value_vars=["tail_pos", "tail_neg"],
                var_name="Tail",
                value_name="Percent",
            )
            fig = (
                px.bar(
                    plot_df,
                    x="index",
                    y="Percent",
                    color="Tail",
                    barmode="group",
                    labels={"index": "Factor"},
                    title=f"Fraction of Assets with |Exposure| > 2 on {last_date.date()}",
                )
                .update_layout(yaxis_tickformat=".0%", yaxis_title="% of Universe")
                .add_hline(y=0, line_color="black")
            )
            html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                               config={"responsive": True, "displayModeBar": False})
            art_id = self._chart_to_artifact(html, "tail_exposure")
            slide = self._get_slide(presentation, 2)
            self._patch_slide(
                slide,
                title="Tail Exposure",
                artifact_id=art_id,
                note=f"Tail exposures at {last_date}",
            )

        # 4) Exposure-correlation matrix ----------------------------------------
        if self.configuration.calculate_exposure_correlation_matrix:
            corr = pfa.correlation_matrix(last_date)
            fig = px.imshow(
                corr,
                labels=dict(x="Asset", y="Asset"),
                title=f"Exposure Correlation Matrix on {last_date.date()}",
            )
            html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                               config={"responsive": True, "displayModeBar": False})
            art_id = self._chart_to_artifact(html, "corr_matrix")
            slide = self._get_slide(presentation, 3)
            self._patch_slide(
                slide,
                title="Exposure Correlation Matrix",
                artifact_id=art_id,
                note=f"Matrix calculated at {last_date}",
            )

        # 5) Scenario-shock analyses --------------------------------------------
        for i, shock_conf in enumerate(self.configuration.shocks_configuration, start=4):
            shock = pd.Series(1.0, index=contrib.columns)
            shock.loc[shock_conf.factor_name] += shock_conf.factor_shock_multiplier
            impact = pfa.scenario_analysis(shock, last_date).reset_index(name="PnlImpact")
            fig = px.bar(impact, x="index", y="PnlImpact",
                         title=f"Hypothetical Shock of 1%: {shock_conf.factor_name}",
                         )
            # 1) Axis: no scaling, just two decimals + % suffix
            fig.update_yaxes(tickformat=".2f", ticksuffix="%")

            # 2) Bar labels: show the raw y value with a % sign
            fig.update_traces(
                texttemplate="%{y:.2f}%",
                textposition="outside"
            )
            html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                               config={"responsive": True, "displayModeBar": False})
            art_id = self._chart_to_artifact(html, f"shock_{shock_conf.factor_name}")
            slide = self._get_slide(presentation, i)
            self._patch_slide(
                slide,
                title=f"Shock: {shock_conf.factor_name}",
                artifact_id=art_id,
                note=f"Impact calculated at {last_date}",
            )




        return None
