"""
=========================================================
Time Series Learning Lab (TSLL)
Trend
=========================================================

This module demonstrates how trends can be represented
using linear, parabolic, and moving-average models.

Students can:
    - Change the slope of a linear trend.
    - Change the curvature of a parabolic trend.
    - Adjust the moving-average window.
    - Compare different trend models visually.
    - Understand the role of a time index.
"""

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dash import html, dcc, Input, Output


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/trend",
    name="Trend",
)


# =========================================================
# Predefined Dataset
# =========================================================

# Monthly sales data with an underlying upward trend
# and some fluctuations.

months = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec",
]

time = np.arange(len(months))

# Fixed fluctuations make the example reproducible.
noise = np.array([
    2, -4, 5, -3, 6, -5,
    4, -2, 7, -6, 3, -1,
    5, -4, 2, -3, 6, -5,
    3, -2, 4, -6, 5, -1,
])

# The default data is generated using a linear trend.
BASE_SLOPE = 3.0
BASE_CURVATURE = 0.0
BASE_INTERCEPT = 100.0


def generate_data(slope, curvature):
    """
    Generate a time series using a linear and quadratic trend.

    Formula:
        Sales = intercept + slope * t + curvature * t^2 + noise

    When curvature = 0:
        The trend is linear.

    When curvature != 0:
        The trend is parabolic.
    """

    sales = (
        BASE_INTERCEPT
        + slope * time
        + curvature * (time ** 2)
        + noise
    )

    return pd.DataFrame(
        {
            "Month": months,
            "Time": time,
            "Sales": sales,
        }
    )


# =========================================================
# Trend Calculation Functions
# =========================================================

def calculate_linear_trend(data):
    """
    Fit a linear regression trend using the time index.
    """

    coefficients = np.polyfit(
        data["Time"],
        data["Sales"],
        deg=1,
    )

    predicted = np.polyval(
        coefficients,
        data["Time"],
    )

    return predicted, coefficients


def calculate_parabolic_trend(data):
    """
    Fit a quadratic regression trend.

    The quadratic model includes:
        Time
        Time squared
    """

    coefficients = np.polyfit(
        data["Time"],
        data["Sales"],
        deg=2,
    )

    predicted = np.polyval(
        coefficients,
        data["Time"],
    )

    return predicted, coefficients


def calculate_moving_average(data, window):
    """
    Calculate a centered moving average.

    The minimum number of observations is set to 1,
    so the edges can also display values.
    """

    return (
        data["Sales"]
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .mean()
    )


# =========================================================
# Chart Creation
# =========================================================

def create_trend_figure(
    data,
    show_linear,
    show_parabolic,
    show_moving_average,
    window,
):
    """
    Create an interactive chart showing the original
    observations and the selected trend models.
    """

    fig = go.Figure()

    # -----------------------------------------------------
    # Original Observations
    # -----------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data["Time"],
            y=data["Sales"],
            mode="markers",
            name="Observed Sales",
            marker=dict(
                size=9,
                color="#34495E",
            ),
            customdata=data["Month"],
            hovertemplate=(
                "Month: %{customdata}<br>"
                "Time: %{x}<br>"
                "Sales: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    # -----------------------------------------------------
    # Linear Trend
    # -----------------------------------------------------

    if show_linear:

        linear_values, _ = calculate_linear_trend(data)

        fig.add_trace(
            go.Scatter(
                x=data["Time"],
                y=linear_values,
                mode="lines",
                name="Linear Trend",
                line=dict(
                    width=3,
                    color="#E74C3C",
                ),
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Linear Trend: %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    # -----------------------------------------------------
    # Parabolic Trend
    # -----------------------------------------------------

    if show_parabolic:

        parabolic_values, _ = calculate_parabolic_trend(data)

        fig.add_trace(
            go.Scatter(
                x=data["Time"],
                y=parabolic_values,
                mode="lines",
                name="Parabolic Trend",
                line=dict(
                    width=3,
                    color="#8E44AD",
                ),
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Parabolic Trend: %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    # -----------------------------------------------------
    # Moving Average
    # -----------------------------------------------------

    if show_moving_average:

        ma_values = calculate_moving_average(
            data,
            window,
        )

        fig.add_trace(
            go.Scatter(
                x=data["Time"],
                y=ma_values,
                mode="lines",
                name=f"Moving Average ({window})",
                line=dict(
                    width=3,
                    color="#27AE60",
                    dash="dash",
                ),
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Moving Average: %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.update_layout(
        title="Comparing Trend Models",

        xaxis_title="Time Index (t)",

        yaxis_title="Sales",

        template="plotly_white",

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),

        margin=dict(
            l=50,
            r=30,
            t=90,
            b=50,
        ),

        height=520,
    )

    # Display month labels instead of numeric time indices.
    tick_positions = list(range(0, len(months), 2))

    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_positions,
        ticktext=[
            f"{months[i]} {i // 12 + 1}"
            for i in tick_positions
        ],
    )

    return fig


# =========================================================
# Trend Explanation
# =========================================================

def create_trend_explanation(slope, curvature, window):
    """
    Generate an explanation based on the selected parameters.
    """

    if curvature > 0:
        curvature_text = (
            "The positive curvature creates an upward-bending "
            "parabolic component. The trend's rate of increase "
            "becomes larger over time."
        )

    elif curvature < 0:
        curvature_text = (
            "The negative curvature creates a downward-bending "
            "parabolic component. The trend's rate of increase "
            "becomes smaller over time and may eventually turn "
            "downward."
        )

    else:
        curvature_text = (
            "The curvature is zero, so the generated trend "
            "has a constant slope and is linear."
        )

    if slope > 0:
        slope_text = "The slope is positive, indicating an upward trend."

    elif slope < 0:
        slope_text = "The slope is negative, indicating a downward trend."

    else:
        slope_text = (
            "The slope is zero, so there is no linear increase "
            "or decrease from the slope component."
        )

    return html.Div(
        [
            html.P(slope_text),
            html.P(curvature_text),
            html.P(
                f"The moving-average window is {window}. "
                "A larger window generally produces a smoother "
                "series but can hide short-term changes."
            ),
        ],
        className="explanation",
    )


# =========================================================
# Python Code Generator
# =========================================================

def create_python_code(slope, curvature, window):
    """
    Generate Python code showing how to calculate trend
    components and a moving average.
    """

    return f'''import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a time index
df["Time"] = np.arange(len(df))

# Linear Trend
linear_model = LinearRegression()
linear_model.fit(df[["Time"]], df["Sales"])

df["Linear_Trend"] = linear_model.predict(
    df[["Time"]]
)

# Parabolic Trend
df["Time_Squared"] = df["Time"] ** 2

parabolic_model = LinearRegression()
parabolic_model.fit(
    df[["Time", "Time_Squared"]],
    df["Sales"]
)

df["Parabolic_Trend"] = parabolic_model.predict(
    df[["Time", "Time_Squared"]]
)

# Moving Average
df["Moving_Average"] = (
    df["Sales"]
    .rolling(window={window}, center=True, min_periods=1)
    .mean()
)

# Display results
print(df[[
    "Time",
    "Sales",
    "Linear_Trend",
    "Parabolic_Trend",
    "Moving_Average"
]])
'''


# =========================================================
# Layout
# =========================================================

layout = dbc.Container(
    [
        # -------------------------------------------------
        # Page Title
        # -------------------------------------------------

        html.H2(
            "📉 Trend",
            className="page-title",
        ),

        html.Hr(),

        # -------------------------------------------------
        # Introduction
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("What Is a Trend?"),

                    html.P(
                        """
                        A trend represents the long-term direction
                        of a time series. It describes whether
                        the values generally increase, decrease,
                        or remain relatively stable over time.
                        """
                    ),

                    html.P(
                        """
                        A trend is different from short-term
                        fluctuations. For example, monthly sales
                        can fluctuate from one month to another
                        while still following an upward trend.
                        """
                    ),

                    html.Div(
                        "Sales(t) = Trend(t) + Short-term fluctuations",
                        className="formula",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Interactive Controls
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Explore Trend Parameters",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Change the parameters below to see how
                        the generated sales data and fitted trend
                        models respond.
                        """
                    ),

                    dbc.Row(
                        [
                            # Slope

                            dbc.Col(
                                [
                                    html.Label(
                                        "Linear Slope",
                                        className="fw-bold",
                                    ),

                                    dcc.Slider(
                                        id="trend-slope",
                                        min=-5,
                                        max=10,
                                        step=0.5,
                                        value=BASE_SLOPE,
                                        marks={
                                            -5: "-5",
                                            0: "0",
                                            5: "5",
                                            10: "10",
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),

                                    html.Small(
                                        "Controls the direction "
                                        "and rate of linear change.",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-4",
                            ),

                            # Curvature

                            dbc.Col(
                                [
                                    html.Label(
                                        "Curvature",
                                        className="fw-bold",
                                    ),

                                    dcc.Slider(
                                        id="trend-curvature",
                                        min=-0.3,
                                        max=0.3,
                                        step=0.05,
                                        value=BASE_CURVATURE,
                                        marks={
                                            -0.3: "-0.3",
                                            0: "0",
                                            0.3: "0.3",
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": True,
                                        },
                                    ),

                                    html.Small(
                                        "Controls the bending "
                                        "of the generated trend.",
                                        className="text-muted",
                                    ),
                                ],
                                md=6,
                                className="mb-4",
                            ),
                        ]
                    ),

                    # Moving Average Window

                    html.Label(
                        "Moving Average Window",
                        className="fw-bold",
                    ),

                    dcc.Slider(
                        id="trend-window",
                        min=2,
                        max=12,
                        step=1,
                        value=4,
                        marks={
                            2: "2",
                            4: "4",
                            6: "6",
                            8: "8",
                            10: "10",
                            12: "12",
                        },
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                    ),

                    html.Hr(),

                    html.Label(
                        "Trend Lines",
                        className="fw-bold",
                    ),

                    dbc.Checklist(
                        id="trend-model-selector",

                        options=[
                            {
                                "label": "Linear Trend",
                                "value": "linear",
                            },
                            {
                                "label": "Parabolic Trend",
                                "value": "parabolic",
                            },
                            {
                                "label": "Moving Average",
                                "value": "moving_average",
                            },
                        ],

                        value=[
                            "linear",
                            "parabolic",
                            "moving_average",
                        ],

                        inline=True,

                        switch=True,
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Main Chart
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Trend Visualization",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Compare the original sales observations
                        with the selected trend models.
                        """
                    ),

                    dcc.Graph(
                        id="trend-graph",
                        config={
                            "displayModeBar": True,
                        },
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Dynamic Explanation
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Understanding the Current Trend",
                        className="section-title",
                    ),

                    html.Div(
                        id="trend-explanation",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Time Dummy
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Time Dummy: Representing Time as a Feature",
                        className="section-title",
                    ),

                    html.P(
                        """
                        A time dummy is a numeric variable that
                        represents the position of an observation
                        in time. In trend modeling, we often use
                        a simple time index such as 0, 1, 2, 3, ...
                        """
                    ),

                    html.Div(
                        "t = 0, 1, 2, 3, ..., n - 1",
                        className="formula",
                    ),

                    html.P(
                        """
                        For example, if we have monthly sales
                        data, the time index increases by one
                        for every month. We can use this variable
                        as an input feature in a regression model
                        to estimate the trend.
                        """
                    ),

                    html.P(
                        """
                        For seasonal patterns, we can also create
                        calendar dummy variables, such as indicators
                        for individual months. These are different
                        from the simple time index used to represent
                        a trend.
                        """
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Linear vs Parabolic Trend
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Linear Trend vs. Parabolic Trend",
                        className="section-title",
                    ),

                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Feature"),
                                        html.Th("Linear Trend"),
                                        html.Th("Parabolic Trend"),
                                    ]
                                )
                            ),

                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td("Equation"),
                                            html.Td("y = a + bt"),
                                            html.Td("y = a + bt + ct²"),
                                        ]
                                    ),

                                    html.Tr(
                                        [
                                            html.Td("Shape"),
                                            html.Td("Straight line"),
                                            html.Td("Parabola"),
                                        ]
                                    ),

                                    html.Tr(
                                        [
                                            html.Td("Slope"),
                                            html.Td("Constant"),
                                            html.Td("Changes over time"),
                                        ]
                                    ),

                                    html.Tr(
                                        [
                                            html.Td("Use case"),
                                            html.Td(
                                                "Approximately constant "
                                                "rate of change"
                                            ),
                                            html.Td(
                                                "Accelerating or "
                                                "decelerating trends"
                                            ),
                                        ]
                                    ),

                                    html.Tr(
                                        [
                                            html.Td("Additional feature"),
                                            html.Td("Time"),
                                            html.Td("Time and Time²"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        bordered=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                    ),

                    html.P(
                        """
                        The quadratic coefficient c controls
                        the curvature. If c is positive, the
                        parabola bends upward. If c is negative,
                        it bends downward. If c is zero, the
                        quadratic model reduces to a linear model.
                        """
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Moving Average Trend
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Moving Average as a Trend Tool",
                        className="section-title",
                    ),

                    html.P(
                        """
                        A moving average smooths short-term
                        fluctuations by averaging nearby
                        observations. This can make the underlying
                        trend easier to see.
                        """
                    ),

                    html.P(
                        """
                        Increasing the window generally creates
                        a smoother curve, but it can also hide
                        short-term changes and make turning points
                        appear later.
                        """
                    ),

                    dbc.Alert(
                        """
                        Remember: A moving average is a smoothing
                        technique, while a linear or parabolic
                        trend is a mathematical model fitted to
                        the data. They are related but not identical.
                        """,
                        color="info",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Python Code
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Python Implementation",
                        className="section-title",
                    ),

                    html.P(
                        """
                        The following code demonstrates how to
                        fit linear and parabolic trends and
                        calculate a moving average.
                        """
                    ),

                    html.Pre(
                        id="trend-code",
                        className="code-block",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Key Takeaways
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "🎯 Key Takeaways",
                        className="section-title",
                    ),

                    html.Ul(
                        [
                            html.Li(
                                "A trend describes the general "
                                "direction of a time series."
                            ),

                            html.Li(
                                "A linear trend assumes a "
                                "constant rate of change."
                            ),

                            html.Li(
                                "A parabolic trend captures "
                                "curvature and changing rates "
                                "of change."
                            ),

                            html.Li(
                                "A time index can be used as "
                                "a feature to model a trend."
                            ),

                            html.Li(
                                "A moving average smooths "
                                "short-term fluctuations."
                            ),

                            html.Li(
                                "A more complex trend model "
                                "is not automatically better. "
                                "Compare models using validation "
                                "data and forecasting performance."
                            ),
                        ]
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),
    ],
    fluid=True,
)


# =========================================================
# Callback
# =========================================================

@dash.callback(
    Output("trend-graph", "figure"),
    Output("trend-explanation", "children"),
    Output("trend-code", "children"),

    Input("trend-slope", "value"),
    Input("trend-curvature", "value"),
    Input("trend-window", "value"),
    Input("trend-model-selector", "value"),
)
def update_trend_dashboard(
    slope,
    curvature,
    window,
    selected_models,
):

    # Generate the dataset using the selected parameters.

    data = generate_data(
        slope=slope,
        curvature=curvature,
    )

    selected_models = selected_models or []

    # Create the visualization.

    figure = create_trend_figure(
        data=data,
        show_linear="linear" in selected_models,
        show_parabolic="parabolic" in selected_models,
        show_moving_average="moving_average" in selected_models,
        window=window,
    )

    # Create the explanation.

    explanation = create_trend_explanation(
        slope=slope,
        curvature=curvature,
        window=window,
    )

    # Generate the Python code.

    code = create_python_code(
        slope=slope,
        curvature=curvature,
        window=window,
    )

    return figure, explanation, code