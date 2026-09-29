"""
=========================================================
Formula Components
=========================================================
"""

from dash import html


def create_formula(title, formula):

    return html.Div(

        [

            html.H5(title),

            html.Pre(

                formula,

                className="formula",

            ),

        ]

    )