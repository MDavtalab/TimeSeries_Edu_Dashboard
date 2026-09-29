"""
=========================================================
Moving Average
Time Series Learning Lab (TSLL)
=========================================================
"""

import dash

from dash import html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.graph_objects as go


# -------------------------------------------------------
# Register Page
# -------------------------------------------------------

dash.register_page(
    __name__,
    path="/moving-average",
    name="Moving Average",
)


# -------------------------------------------------------
# Sample Dataset
# -------------------------------------------------------

df = pd.DataFrame(
    {
        "Month": [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ],
        "Sales": [
            100, 120, 110, 140, 160, 150,
            180, 200, 190, 220, 240, 230,
        ],
    }
)

df["Month_Index"] = range(len(df))


# -------------------------------------------------------
# Initial Settings
# -------------------------------------------------------

DEFAULT_WINDOW = 3
DEFAULT_CENTER = True
DEFAULT_MIN_PERIODS = 1
DEFAULT_STEP = 5


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def calculate_moving_average(
    data,
    window,
    center,
    min_periods,
):
    """
    Calculate the moving average using Pandas.
    """

    return data["Sales"].rolling(
        window=window,
        center=center,
        min_periods=min_periods,
    ).mean()


def get_window_bounds(
    step,
    window,
    center,
    data_length,
):
    """
    Get the observation indexes used by the selected
    rolling window.

    For center=False, the window ends at the selected step.

    For center=True, the window is centered around the
    selected step.
    """

    if center:

        left = (window - 1) // 2
        right = window // 2

        start = step - left
        end = step + right

    else:

        start = step - window + 1
        end = step

    start = max(0, start)
    end = min(data_length - 1, end)

    return start, end


def create_figure(
    data,
    moving_average,
    step,
    window,
    center,
):
    """
    Create the interactive time series visualization.
    """

    fig = go.Figure()

    # ---------------------------------------------------
    # Sales
    # ---------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data["Month"],
            y=data["Sales"],
            mode="lines+markers",
            name="Sales",
            line=dict(width=3),
            marker=dict(size=8),
        )
    )

    # ---------------------------------------------------
    # Moving Average
    # ---------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data["Month"],
            y=moving_average,
            mode="lines+markers",
            name="Moving Average",
            line=dict(width=3, dash="dash"),
            marker=dict(size=7),
        )
    )

    # ---------------------------------------------------
    # Highlight Active Window
    # ---------------------------------------------------

    start, end = get_window_bounds(
        step=step,
        window=window,
        center=center,
        data_length=len(data),
    )

    fig.add_vrect(
        x0=start - 0.5,
        x1=end + 0.5,
        fillcolor="LightSkyBlue",
        opacity=0.25,
        line_width=0,
        layer="below",
    )

    # ---------------------------------------------------
    # Selected Month
    # ---------------------------------------------------

    fig.add_vline(
        x=step,
        line_dash="dot",
        line_width=2,
        line_color="gray",
    )

    # ---------------------------------------------------
    # Layout
    # ---------------------------------------------------

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(
            l=40,
            r=30,
            t=40,
            b=40,
        ),
        xaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=data["Month_Index"],
            ticktext=data["Month"],
        ),
        yaxis=dict(
            title="Sales",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig


def create_calculation(
    data,
    moving_average,
    step,
    window,
    center,
    min_periods,
):
    """
    Generate the calculation explanation for the
    selected time step.
    """

    start, end = get_window_bounds(
        step=step,
        window=window,
        center=center,
        data_length=len(data),
    )

    selected_data = data.iloc[start:end + 1]

    values = selected_data["Sales"].tolist()

    valid_values = len(values)

    ma_value = moving_average.iloc[step]

    month = data.iloc[step]["Month"]

    # ---------------------------------------------------
    # Calculation Formula
    # ---------------------------------------------------

    if pd.isna(ma_value):

        formula = (
            f"Moving Average for {month}\n\n"
            f"Not enough observations.\n"
            f"Required minimum periods: {min_periods}\n"
            f"Available observations: {valid_values}"
        )

    else:

        values_text = " + ".join(
            f"{value:.0f}" for value in values
        )

        formula = (
            f"Moving Average for {month}\n\n"
            f"({values_text}) / {valid_values}\n\n"
            f"= {ma_value:.2f}"
        )

    return formula


def create_python_code(
    window,
    center,
    min_periods,
):
    """
    Generate equivalent Pandas code.
    """

    return (
        "df['MA'] = df['Sales'].rolling(\n"
        f"    window={window},\n"
        f"    center={center},\n"
        f"    min_periods={min_periods},\n"
        ").mean()"
    )


# -------------------------------------------------------
# Layout
# -------------------------------------------------------

layout = dbc.Container(
    [

        # ===============================================
        # Page Header
        # ===============================================

        html.H2(
            "📈 Moving Average",
            className="page-title",
        ),

        html.Hr(),

        html.P(
            """
            Explore how moving averages smooth time series
            data. Adjust the window size, alignment, and
            minimum number of observations to see how the
            results change.
            """,
            className="explanation",
        ),

        # ===============================================
        # Controls
        # ===============================================

        dbc.Card(
            [

                dbc.CardHeader(
                    html.H5(
                        "⚙️ Controls",
                        className="mb-0",
                    )
                ),

                dbc.CardBody(
                    [

                        # Window Size

                        html.Label(
                            "Window Size",
                            className="fw-bold",
                        ),

                        dcc.Slider(
                            id="ma-window",
                            min=2,
                            max=12,
                            step=1,
                            value=DEFAULT_WINDOW,
                            marks={
                                i: str(i)
                                for i in range(2, 13)
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                        html.Br(),

                        # Center Checkbox

                        dbc.Checklist(
                            id="ma-center",
                            options=[
                                {
                                    "label": "Center the window",
                                    "value": "center",
                                }
                            ],
                            value=(
                                ["center"]
                                if DEFAULT_CENTER
                                else []
                            ),
                            switch=True,
                        ),

                        html.Br(),

                        # Minimum Periods

                        html.Label(
                            "Minimum Periods",
                            className="fw-bold",
                        ),

                        dcc.Slider(
                            id="ma-min-periods",
                            min=1,
                            max=12,
                            step=1,
                            value=DEFAULT_MIN_PERIODS,
                            marks={
                                i: str(i)
                                for i in range(1, 13)
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                        html.Br(),

                        # Selected Time Step

                        html.Label(
                            "Selected Month",
                            className="fw-bold",
                        ),

                        dcc.Slider(
                            id="ma-step",
                            min=0,
                            max=len(df) - 1,
                            step=1,
                            value=DEFAULT_STEP,
                            marks={
                                i: df.iloc[i]["Month"]
                                for i in range(len(df))
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),

                    ]
                ),

            ],
            className="ts-card",
        ),

        # ===============================================
        # Visualization
        # ===============================================

        dbc.Card(
            [

                dbc.CardHeader(
                    html.H5(
                        "📊 Sales and Moving Average",
                        className="mb-0",
                    )
                ),

                dbc.CardBody(
                    dcc.Graph(
                        id="ma-graph",
                        config={
                            "displayModeBar": True,
                            "responsive": True,
                        },
                    )
                ),

            ],
            className="ts-card",
        ),

        # ===============================================
        # Calculation and Formula
        # ===============================================

        dbc.Row(
            [

                # Calculation Panel

                dbc.Col(
                    dbc.Card(
                        [

                            dbc.CardHeader(
                                html.H5(
                                    "🧮 Calculation",
                                    className="mb-0",
                                )
                            ),

                            dbc.CardBody(
                                html.Pre(
                                    id="ma-calculation",
                                    className="formula",
                                )
                            ),

                        ],
                        className="ts-card",
                    ),
                    md=6,
                ),

                # Python Code Panel

                dbc.Col(
                    dbc.Card(
                        [

                            dbc.CardHeader(
                                html.H5(
                                    "🐍 Equivalent Python Code",
                                    className="mb-0",
                                )
                            ),

                            dbc.CardBody(
                                html.Pre(
                                    id="ma-code",
                                    className="code-block",
                                )
                            ),

                        ],
                        className="ts-card",
                    ),
                    md=6,
                ),

            ]
        ),

        # ===============================================
        # Dataset Table
        # ===============================================

        dbc.Card(
            [

                dbc.CardHeader(
                    html.H5(
                        "📋 Dataset",
                        className="mb-0",
                    )
                ),

                dbc.CardBody(
                    dash_table.DataTable(
                        id="ma-table",
                        columns=[
                            {
                                "name": "Month",
                                "id": "Month",
                            },
                            {
                                "name": "Sales",
                                "id": "Sales",
                            },
                            {
                                "name": "Moving Average",
                                "id": "Moving_Average",
                            },
                        ],
                        data=[],
                        page_size=12,
                        style_table={
                            "overflowX": "auto",
                        },
                        style_cell={
                            "textAlign": "center",
                            "padding": "10px",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#3498DB",
                            "color": "white",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": (
                                        "{Month_Index} = "
                                        "{ma-step}"
                                    )
                                },
                                "backgroundColor": "#D6EAF8",
                            }
                        ],
                    )
                ),

            ],
            className="ts-card",
        ),

    ],
    fluid=True,
)


# -------------------------------------------------------
# Callback
# -------------------------------------------------------

@dash.callback(
    Output("ma-graph", "figure"),
    Output("ma-calculation", "children"),
    Output("ma-code", "children"),
    Output("ma-table", "data"),

    Input("ma-window", "value"),
    Input("ma-center", "value"),
    Input("ma-min-periods", "value"),
    Input("ma-step", "value"),
)
def update_moving_average(
    window,
    center_value,
    min_periods,
    step,
):
    """
    Update the visualization, calculation, code,
    and table whenever a control changes.
    """

    # ---------------------------------------------------
    # Validate Inputs
    # ---------------------------------------------------

    center = "center" in center_value

    min_periods = min(min_periods, window)

    # ---------------------------------------------------
    # Calculate Moving Average
    # ---------------------------------------------------

    moving_average = calculate_moving_average(
        data=df,
        window=window,
        center=center,
        min_periods=min_periods,
    )

    # ---------------------------------------------------
    # Create Figure
    # ---------------------------------------------------

    fig = create_figure(
        data=df,
        moving_average=moving_average,
        step=step,
        window=window,
        center=center,
    )

    # ---------------------------------------------------
    # Create Calculation
    # ---------------------------------------------------

    calculation = create_calculation(
        data=df,
        moving_average=moving_average,
        step=step,
        window=window,
        center=center,
        min_periods=min_periods,
    )

    # ---------------------------------------------------
    # Generate Python Code
    # ---------------------------------------------------

    code = create_python_code(
        window=window,
        center=center,
        min_periods=min_periods,
    )

    # ---------------------------------------------------
    # Create Table
    # ---------------------------------------------------

    table_df = df.copy()

    table_df["Moving_Average"] = moving_average.round(2)

    table_data = table_df[
        ["Month", "Sales", "Moving_Average"]
    ].to_dict("records")

    return (
        fig,
        calculation,
        code,
        table_data,
    )