import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/time-series-features",
    name="Time Series as Features",
)

# =========================================================
# 1. DATA GENERATION
# =========================================================

def generate_data(mode="time", n=100, seed=42):
    """
    Generate simple time-series data for three cases:
    - time: trend only
    - serial: AR(1) process
    - combined: trend + serial dependence
    """

    rng = np.random.default_rng(seed)

    t = np.arange(n)
    noise = rng.normal(0, 2, n)

    if mode == "time":
        # Time dependence: the value increases with time
        y = 10 + 0.5 * t + noise

    elif mode == "serial":
        # Serial dependence: each value depends on the previous value
        y = np.zeros(n)
        y[0] = 10

        for i in range(1, n):
            y[i] = 0.8 * y[i - 1] + noise[i]

    else:
        # Combined: trend + serial dependence
        y = np.zeros(n)
        y[0] = 10

        for i in range(1, n):
            trend = 0.3 * i
            y[i] = trend + 0.6 * (y[i - 1] - 0.3 * (i - 1)) + noise[i]

    df = pd.DataFrame({
        "Time": t,
        "Value": y
    })

    return df


# =========================================================
# 2. FEATURE CREATION
# =========================================================

def create_features(df, mode="time", lag=1):
    """
    Create features based on the selected type of dependence.
    """

    data = df.copy()

    if mode == "time":
        data["Feature"] = data["Time"]
        feature_name = "Time index (t)"

    elif mode == "serial":
        data["Feature"] = data["Value"].shift(lag)
        feature_name = f"Lag {lag} (y[t-{lag}])"

    else:
        data["Feature"] = data["Time"]
        data["Lag"] = data["Value"].shift(lag)
        feature_name = f"Time index + Lag {lag}"

    data = data.dropna().reset_index(drop=True)

    return data, feature_name


# =========================================================
# 3. PAGE LAYOUT
# =========================================================

layout = dbc.Container(
    [

        # -------------------------------------------------
        # PAGE TITLE
        # -------------------------------------------------

        html.H2(
            "Time Series as Features",
            className="mt-4 mb-2"
        ),

        html.P(
            "Explore how time-based features and past observations "
            "help a model understand a time series.",
            className="text-muted mb-4"
        ),

        # -------------------------------------------------
        # CONTROLS
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.Label(
                        "Select a dependence type",
                        className="fw-bold"
                    ),

                    dcc.Dropdown(
                        id="ts-feature-mode",
                        options=[
                            {
                                "label": "Time Dependence (Trend)",
                                "value": "time"
                            },
                            {
                                "label": "Serial Dependence (Lag)",
                                "value": "serial"
                            },
                            {
                                "label": "Combined (Time + Lag)",
                                "value": "combined"
                            }
                        ],
                        value="time",
                        clearable=False
                    ),

                    html.Br(),

                    html.Label(
                        "Select lag",
                        className="fw-bold"
                    ),

                    dcc.Slider(
                        id="ts-feature-lag",
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        marks={
                            i: str(i)
                            for i in range(1, 11)
                        },
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True
                        }
                    ),

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "What are we learning?",
                        className="mb-3"
                    ),

                    html.Div(
                        id="ts-feature-explanation"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # TIME SERIES CHART
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "1. Time-Series Plot",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="ts-feature-series"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # FEATURE RELATIONSHIP CHART
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "2. Feature vs. Target",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="ts-feature-scatter"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # KEY TAKEAWAY
        # -------------------------------------------------

        dbc.Alert(
            [
                html.H5("Key Takeaway"),

                html.P(
                    "Time dependence uses the time index as a feature. "
                    "Serial dependence uses previous target values "
                    "as features."
                ),

                html.P(
                    "Both approaches can be used together in the "
                    "same forecasting model.",
                    className="mb-0"
                )
            ],
            color="info",
            className="mt-3 mb-4"
        ),

    ],
    fluid=True
)


# =========================================================
# 4. CALLBACK
# =========================================================

@__import__("dash").callback(
    Output("ts-feature-series", "figure"),
    Output("ts-feature-scatter", "figure"),
    Output("ts-feature-explanation", "children"),

    Input("ts-feature-mode", "value"),
    Input("ts-feature-lag", "value")
)
def update_dashboard(mode, lag):

    # Generate data
    df = generate_data(mode=mode)

    # Create features
    data, feature_name = create_features(
        df,
        mode=mode,
        lag=lag
    )

    # -----------------------------------------------------
    # TIME SERIES CHART
    # -----------------------------------------------------

    fig_series = go.Figure()

    fig_series.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Observed values",
            line=dict(width=2)
        )
    )

    fig_series.update_layout(
        title="How the values change over time",
        xaxis_title="Time step",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified"
    )

    # -----------------------------------------------------
    # FEATURE VS TARGET CHART
    # -----------------------------------------------------

    fig_scatter = go.Figure()

    if mode == "time":

        fig_scatter.add_trace(
            go.Scatter(
                x=data["Feature"],
                y=data["Value"],
                mode="markers",
                name="Observations"
            )
        )

        fig_scatter.update_layout(
            title="Time index vs. current value",
            xaxis_title="Time index (t)",
            yaxis_title="Current value"
        )

        explanation = html.Div(
            [

                html.P(
                    "Here, the feature is the time index."
                ),

                html.Code(
                    "X = df['Time']"
                ),

                html.P(
                    "The model learns how the target changes "
                    "as time moves forward."
                ),

                html.P(
                    "This is useful for learning a trend, "
                    "such as sales increasing over time."
                )

            ]
        )

    elif mode == "serial":

        fig_scatter.add_trace(
            go.Scatter(
                x=data["Feature"],
                y=data["Value"],
                mode="markers",
                name="Observations"
            )
        )

        fig_scatter.update_layout(
            title=f"Lag {lag} vs. current value",
            xaxis_title=f"Previous value (y[t-{lag}])",
            yaxis_title="Current value (y[t])"
        )

        explanation = html.Div(
            [

                html.P(
                    "Here, the feature is a previous observation."
                ),

                html.Code(
                    f"X = df['Value'].shift({lag})"
                ),

                html.P(
                    f"The model uses the value from {lag} time "
                    f"step(s) earlier to predict the current value."
                ),

                html.P(
                    "If past values contain useful information "
                    "about the current value, the series has "
                    "serial dependence."
                )

            ]
        )

    else:

        fig_scatter.add_trace(
            go.Scatter(
                x=data["Time"],
                y=data["Value"],
                mode="markers",
                name="Time feature",
                marker=dict(symbol="circle")
            )
        )

        fig_scatter.update_layout(
            title="Time index vs. current value",
            xaxis_title="Time index (t)",
            yaxis_title="Current value (y[t])"
        )

        explanation = html.Div(
            [

                html.P(
                    "This example combines a time-based feature "
                    "and a lag-based feature."
                ),

                html.Code(
                    f"X = df[['Time', 'Value']].assign("
                    f"Lag=df['Value'].shift({lag}))"
                ),

                html.P(
                    "The model can use both the passage of time "
                    "and past observations."
                ),

                html.P(
                    "This is the basic idea behind combining "
                    "time features with serial-dependence features."
                )

            ]
        )

    fig_scatter.update_layout(
        template="plotly_white",
        hovermode="closest"
    )

    return fig_series, fig_scatter, explanation