"""
=========================================================
Home Page
=========================================================
"""

import dash

from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/",
    name="Home",
)


layout = dbc.Container(

    [

        html.H1(
            "📈 Time Series Learning Lab",
            className="page-title",
        ),

        html.Hr(),

        html.P(

            """
            Welcome to the Time Series Learning Lab!

            This application is designed to help students learn
            Time Series Analysis through interactive visualizations,
            animations, and hands-on experiments.
            """,

            className="explanation",

        ),

        html.Br(),

        dbc.Alert(

            "👈 Select a topic from the left sidebar to begin learning.",

            color="primary",

        ),

    ],

    fluid=True,

)