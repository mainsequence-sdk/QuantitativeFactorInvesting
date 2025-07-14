import pytz
import json
from factorinvesting.apps.factor_esposure_app import (FactorExposureApp,FactorAnalysisConfiguration,
                                                      ShockAnalysisConfiguration)
import mainsequence.client as ms_client
import datetime
if __name__ == "__main__":

    PORTFOLIO_TICKER="portfo446B"#"portfo446B" #"portfo505B"


    folder=ms_client.Folder.filter()[-1]
    # presentation = ms_client.Presentation.get_or_create_by_title(title="asdfasdfasdf",
    #                                                     folder=folder.id,
    #                                                     description="Test Presentation"
    #                                                     )
    # presentation.add_slide()
    # target_slide=presentation.slides[0]
    # target_body=json.loads(target_slide.body)
    # target_body["content"].append(ms_client.TextParagraph.paragraph(
    #     "This is the text of my body",
    #     text_align=None
    # ).model_dump()
    # )
    # target_body["content"].append(json.loads(ms_client.AppNode.make_app_node(id=50).model_dump_json())
    #                               )
    #
    # target_body["content"].append(
    #     ms_client.TextParagraph.heading(
    #         "Section Title",
    #         level=2,
    #         text_align="center"
    #     ).model_dump()
    # )
    # target_slide.patch(body=json.dumps(target_body))


    shocks=[ShockAnalysisConfiguration(factor_name="beta",
                                       factor_shock_multiplier=.05
                                       )]
    app_configuration=FactorAnalysisConfiguration(
        portfolio_ticker=PORTFOLIO_TICKER,
        start_date=datetime.datetime(2020, 1, 1,tzinfo=pytz.utc),
        end_date=datetime.datetime(2025, 1, 1,tzinfo=pytz.utc),
        shocks_configuration=shocks,

    )

    app=FactorExposureApp(configuration=app_configuration)
    app.run()




