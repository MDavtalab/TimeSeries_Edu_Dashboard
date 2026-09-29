"""
=========================================================
Footer
=========================================================
"""

import dash_bootstrap_components as dbc
from dash import html

from config import APP_VERSION


footer = dbc.Container(

    dbc.Row(

        dbc.Col(

            html.Div(

                [

                    html.Hr(),

                    html.P(

                        f"Time Series Learning Lab • Version {APP_VERSION}",

                        className="text-center text-muted",

                    ),

                ]

            )

        )

    ),

    fluid=True,

    className="mt-4 mb-2",

)