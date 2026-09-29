"""
=========================================================
Hybrid Models
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

    path="/hybrid-models",
    name="Hybrid Models",

)

# -------------------------------------------------------
# Layout
# -------------------------------------------------------

layout = dbc.Container(

    [

        html.H2(

            "🔄 Hybrid Models",
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