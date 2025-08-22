import datetime
import os
from enum import Enum
from jinja2 import Environment, FileSystemLoader

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from mainsequence.reportbuilder.model import StyleSettings, ThemeMode
from mainsequence.reportbuilder.slide_templates import generic_plotly_table
from mainsequence.virtualfundbuilder.resource_factory.app_factory import (
    HtmlApp,
    register_app,
)
from mainsequence.virtualfundbuilder.utils import get_vfb_logger
import pytz
logger = get_vfb_logger()


class DerivativeType(str, Enum):
    EUROPEAN_CALL = "European Call"
    EUROPEAN_PUT = "European Put"


class DerivativesPricerConfiguration(BaseModel):
    report_title: str = "Derivatives Pricing & Risk Report"
    underlying_asset: str = Field(
        default="MSFT", description="The ticker symbol for the underlying asset."
    )
    derivative_type: DerivativeType = Field(
        default=DerivativeType.EUROPEAN_CALL,
        description="The type of the derivative contract.",
    )
    spot_price: float = Field(
        default=450.0, description="Current market price of the underlying asset.", gt=0
    )
    strike_price: float = Field(
        default=460.0, description="The strike price of the option.", gt=0
    )
    maturity_date: str = Field(
        default="2025-12-30T00:00:00+02:00",
        description="The expiration date of the option.",
    )
    risk_free_rate: float = Field(
        default=0.045,
        description="The risk-free interest rate (e.g., 0.05 for 5%).",
        ge=0,
    )
    volatility: float = Field(
        default=0.22,
        description="The annualized volatility of the underlying asset (e.g., 0.20 for 20%).",
        ge=0,
    )


def generate_plausible_greeks(config: DerivativesPricerConfiguration):
    maturity_as_date = datetime.datetime.fromisoformat(config.maturity_date).date()
    time_to_maturity_years = (
        maturity_as_date - datetime.date.today()
    ).days / 365.25

    if time_to_maturity_years <= 0:
        price = (
            max(0, config.spot_price - config.strike_price)
            if config.derivative_type == DerivativeType.EUROPEAN_CALL
            else max(0, config.strike_price - config.spot_price)
        )
        return {"Price": price, "Delta": 0.0, "Vega": 0.0, "Theta": 0.0}

    if config.derivative_type == DerivativeType.EUROPEAN_CALL:
        delta = np.clip(
            0.45 + (config.spot_price - config.strike_price) * 0.02, 0.05, 0.95
        )
    else:
        delta = np.clip(
            -0.55 + (config.spot_price - config.strike_price) * 0.02, -0.95, -0.05
        )

    theta = (
        -(config.volatility * config.spot_price)
        / (2 * np.sqrt(time_to_maturity_years))
        * 0.01
    )
    vega = (config.spot_price * np.sqrt(time_to_maturity_years)) * 0.1

    intrinsic_value = (
        max(0, config.spot_price - config.strike_price)
        if config.derivative_type == DerivativeType.EUROPEAN_CALL
        else max(0, config.strike_price - config.spot_price)
    )
    time_value = vega * config.volatility * 0.5
    price = intrinsic_value + time_value

    return {"Price": price, "Delta": delta, "Vega": vega, "Theta": theta}


def generate_volatility_smile_chart(config: DerivativesPricerConfiguration) -> str:
    """Generates a Plotly chart for the volatility smile."""
    strikes = np.linspace(config.strike_price * 0.8, config.strike_price * 1.2, 15)
    spot_normalized_strikes = (strikes - config.spot_price) / config.spot_price
    skew = -0.1 * spot_normalized_strikes
    smile = 0.5 * (spot_normalized_strikes**2)
    noise = np.random.normal(0, 0.002, len(strikes))
    vols = config.volatility + skew + smile + noise

    fig = go.Figure(
        data=go.Scatter(x=strikes, y=vols, mode="lines+markers", name="Implied Volatility")
    )
    fig.update_layout(
        title=f"Implied Volatility Smile for {config.underlying_asset}",
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".2%",
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@register_app()
class DerivativesPricerApp(HtmlApp):
    """
    An application for displaying derivatives pricing information.
    It generates a report with model inputs, pricing results (including Greeks),
    and a volatility smile chart.
    """

    configuration_class = DerivativesPricerConfiguration

    def run(self) -> str:
        style = StyleSettings(mode=ThemeMode.light)
        chart_font_family = style.font_family_paragraphs
        chart_label_font_size = style.chart_label_font_size

        # --- 1. Generate Data ---
        base_results = generate_plausible_greeks(self.configuration)
        maturity_as_date = datetime.datetime.fromisoformat(self.configuration.maturity_date).date()
        time_to_maturity_days = (
            maturity_as_date - datetime.date.today()
        ).days

        # --- 2. Generate HTML Components ---
        inputs_headers = ["Parameter", "Value"]
        inputs_rows = [
            ["Underlying Asset", self.configuration.underlying_asset],
            ["Spot Price", f"${self.configuration.spot_price:.2f}"],
            ["Strike Price", f"${self.configuration.strike_price:.2f}"],
            ["Maturity Date", maturity_as_date.strftime("%Y-%m-%d")],
            ["Days to Maturity", f"{time_to_maturity_days}"],
            ["Volatility (Implied)", f"{self.configuration.volatility:.2%}"],
        ]

        inputs_table_html = generic_plotly_table(
            headers=inputs_headers,
            rows=inputs_rows,
            fig_width=450,
            header_font_dict=dict(
                color=style.background_color, size=12, family=chart_font_family
            ),
            cell_font_dict=dict(
                size=chart_label_font_size,
                family=chart_font_family,
                color=style.paragraph_color,
            ),
            include_plotlyjs="cdn"
        )

        results_headers = ["Metric", "Value"]
        results_rows = [
            ["Price", f"${base_results['Price']:.2f}"],
            ["Delta", f"${base_results['Delta']:.4f}"],
            ["Vega", f"${base_results['Vega']:.4f}"],
            ["Theta", f"${base_results['Theta']:.4f}"],
        ]

        results_table_html = generic_plotly_table(
            headers=results_headers,
            rows=results_rows,
            fig_width=450,
            header_font_dict=dict(
                color=style.background_color, size=12, family=chart_font_family
            ),
            cell_font_dict=dict(
                size=chart_label_font_size,
                family=chart_font_family,
                color=style.paragraph_color,
            ),
            include_plotlyjs="cdn"
        )

        smile_chart_html = generate_volatility_smile_chart(self.configuration)

        # --- 3. Assemble Report Content for Jinja Template ---
        report_content = f"""
        <div class="col-lg-6"><h4 class="mt-4">Contract & Market Inputs</h4>{inputs_table_html}</div>
        <div class="col-lg-6"><h4 class="mt-4">Estimated Pricing & Key Risks</h4>{results_table_html}</div>
        <hr>
        <h3 class="mt-4">Implied Volatility Market Structure</h3>
        <div class="text-center">{smile_chart_html}</div>
        """

        # --- 4. Set up Jinja and Render Template ---
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
        template = env.get_template("report.html")

        template_context = {
            "report_title": self.configuration.report_title,
            "summary": f"Pricing and risk analysis for a {self.configuration.derivative_type.value} on {self.configuration.underlying_asset}.",
            "report_content": report_content,
            "authors": "Quantitative Analytics Desk",
            "sector": "Derivatives",
            "region": "N/A",
            "topics": ["Options Pricing", "Risk Management", "Volatility"],
            "report_id": f"DRV_{self.configuration.underlying_asset}_{datetime.date.today().strftime('%Y%m%d')}",
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "current_year": datetime.datetime.now().year,
            "logo_location": "https://main-sequence.app/static/media/logos/MS_logo_long_white.png",
        }

        return template.render(template_context)


if __name__ == "__main__":
    config = DerivativesPricerConfiguration(
        underlying_asset="AAPL",
        spot_price=190.50,
        strike_price=195.00,
    )
    app = DerivativesPricerApp(config)

    html_report = app.run()