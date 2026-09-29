"""
=========================================================
Time Series Learning Lab (TSLL)
Main Application
=========================================================
"""

import dash
import dash_bootstrap_components as dbc

from config import APP_NAME, THEME
#from config import APP_NAME, THEME

from layouts.navbar import navbar
from layouts.sidebar import sidebar
from layouts.footer import footer


# -------------------------------------------------------
# Dash App
# -------------------------------------------------------

app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[THEME],

)

app.title = APP_NAME
server = app.server

# -------------------------------------------------------
# Layout
# -------------------------------------------------------

app.layout = dbc.Container(

    [
        navbar,
        dbc.Row(
            [
                dbc.Col(
                    sidebar,
                    width=3,
                ),

                dbc.Col(
                    dash.page_container,
                    width=9,
                ),
            ],
            className="mt-3",
        ),
        footer,
    ],
    fluid=True,
)

# -------------------------------------------------------
# Run Server
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)