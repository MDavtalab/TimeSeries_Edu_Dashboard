"""
=========================================================
Navigation Bar
=========================================================
"""

import dash_bootstrap_components as dbc
from dash import html

from config import APP_NAME


navbar = dbc.Navbar(

    dbc.Container(

        [

            dbc.NavbarBrand(

                [

                    html.Span(
                        "📈",
                        className="me-2",
                    ),

                    html.Span(
                        APP_NAME,
                    ),

                ],

                className="fw-bold fs-4",

            ),

        ],

        fluid=True,

    ),

    color="primary",

    dark=True,

)