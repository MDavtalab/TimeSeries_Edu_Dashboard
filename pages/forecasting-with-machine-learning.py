"""
=========================================================
Forecasting with Machine Learning
=========================================================
"""

import dash

from dash import html
import dash_bootstrap_components as dbc


# -------------------------------------------------------
# Register Page
# -------------------------------------------------------

dash.register_page(

    __name__,

    path="/forecasting-with-machine-learning",
    name="Forecasting with Machine Learning",

)

# -------------------------------------------------------
# Layout
# -------------------------------------------------------

layout = dbc.Container(

    [

        html.H2(

            "🔄 Forecasting with Machine Learning",
            className="page-title",

        ),

        html.Hr(),
        dbc.Alert(

            "🚧 This module is currently under development.",
            color="warning",

        ),

        html.P(
            """
            

            """,
            className="explanation",

        ),

    ],
    fluid=True,
)