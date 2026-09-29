"""
=========================================================
Sidebar
=========================================================
"""

import dash_bootstrap_components as dbc
from dash import html

from config import PAGES


sidebar = dbc.Card(

    [

        dbc.CardHeader(

            html.H5(

                "Modules",

                className="mb-0",

            )

        ),

        dbc.CardBody(

            dbc.Nav(

                [

                    dbc.NavLink(

                        [

                            html.Span(
                                page["icon"],
                                className="me-2",
                            ),

                            html.Span(
                                page["title"],
                            ),

                        ],

                        href=page["path"],

                        active="exact",

                        disabled=not page["enabled"],

                    )

                    for page in PAGES

                ],

                vertical=True,

                pills=True,

                className="gap-2",

            )

        ),

    ],

    className="sidebar-card h-100",

)