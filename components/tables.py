"""
=========================================================
Data Tables
=========================================================
"""

from dash import dash_table


def create_table(

    table_id,

    data,

    columns,

):

    return dash_table.DataTable(

        id=table_id,

        data=data,

        columns=columns,

        page_size=15,

        style_table={

            "overflowX": "auto",

        },

        style_cell={

            "textAlign": "center",

            "padding": "8px",

        },

        style_header={

            "fontWeight": "bold",

        },

    )