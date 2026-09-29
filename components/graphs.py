"""
=========================================================
Plotly Graph Components
=========================================================
"""

from dash import dcc
import plotly.graph_objects as go


def create_empty_graph(graph_id):

    fig = go.Figure()

    fig.update_layout(

        template="plotly_white",

        height=450,

        margin=dict(

            l=30,

            r=30,

            t=30,

            b=30,

        ),

    )

    return dcc.Graph(

        id=graph_id,

        figure=fig,

    )