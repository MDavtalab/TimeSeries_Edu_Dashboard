"""
=========================================================
Interactive Controls
=========================================================
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_slider(

    slider_id,

    min_value,

    max_value,

    value,

    step=1,

    label="",

):

    return dbc.Card(

        [

            dbc.CardHeader(label),

            dbc.CardBody(

                dcc.Slider(

                    id=slider_id,

                    min=min_value,

                    max=max_value,

                    step=step,

                    value=value,

                    tooltip={

                        "placement": "bottom",

                        "always_visible": True,

                    },

                )

            ),

        ],

        className="ts-card",

    )


def create_checkbox(

    checkbox_id,

    label,

    value=False,

):

    return dbc.Checklist(

        options=[

            {

                "label": label,

                "value": True,

            }

        ],

        value=[True] if value else [],

        id=checkbox_id,

        switch=True,

    )