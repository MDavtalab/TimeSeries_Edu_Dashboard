"""
=========================================================
Cards
Reusable Bootstrap Cards
=========================================================
"""

import dash_bootstrap_components as dbc


def create_card(title, children):

    """
    Create a standard TSLL card.
    """

    return dbc.Card(

        [

            dbc.CardHeader(

                title,

                className="fw-bold",

            ),

            dbc.CardBody(

                children,

            ),

        ],

        className="ts-card",

    )