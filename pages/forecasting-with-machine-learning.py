import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/forecasting-with-machine-learning",
    name="Forecasting With Machine Learning",
)


# =========================================================
# 1. GENERATE EXAMPLE TIME SERIES
# =========================================================

def generate_data(n=100, seed=42):
    """
    Create a simple time series containing:

    - Trend
    - Seasonality
    - Noise
    """

    rng = np.random.default_rng(seed)

    t = np.arange(n)

    # Trend
    trend = 0.15 * t

    # Seasonal pattern
    seasonal = 5 * np.sin(2 * np.pi * t / 12)

    # Random noise
    noise = rng.normal(0, 1.0, n)

    # Final time series
    y = 20 + trend + seasonal + noise

    return pd.DataFrame(
        {
            "Time": t,
            "Value": y,
        }
    )


# =========================================================
# 2. CREATE NUMERIC MULTI-STEP TRAINING DATA
# =========================================================

def create_multistep_dataset(
    df,
    input_window=6,
    forecast_horizon=3,
):
    """
    Convert the time series into a supervised-learning dataset.

    Example:

        X = [past values]
        y = [future values]

    For example:

        X = [20.3, 21.6, 25.4, 26.4, 23.0, 21.9]

        y = [21.0, 18.2, 16.9]

    Returns
    -------
    X : numpy array
        Past observations.

    y : numpy array
        Future observations.

    origins : numpy array
        Forecast origin for each training example.
    """

    values = df["Value"].to_numpy()

    X = []
    y = []
    origins = []

    number_of_examples = (
        len(values)
        - input_window
        - forecast_horizon
        + 1
    )

    for i in range(number_of_examples):

        # Past observations
        past_values = values[
            i : i + input_window
        ]

        # Future observations
        future_values = values[
            i + input_window :
            i + input_window + forecast_horizon
        ]

        X.append(past_values)
        y.append(future_values)

        # Point where the future begins
        origins.append(i + input_window)

    return (
        np.array(X),
        np.array(y),
        np.array(origins),
    )


# =========================================================
# 3. FORMAT VALUES FOR THE EDUCATIONAL TABLE
# =========================================================

def format_values(values):
    """
    Convert a sequence of numbers into a readable string.

    Example:

        [20.321625426423]

    becomes:

        [20.3, 21.6, 25.4, 26.4, 23.0]
    """

    return (
        "["
        + ", ".join(
            f"{float(value):.1f}"
            for value in values
        )
        + "]"
    )


# =========================================================
# 4. CREATE TABLE DATA
# =========================================================

def create_training_table_data(
    X,
    y,
    number_of_examples=5,
):
    """
    Create a small, human-readable version of the
    training dataset for the dashboard table.

    The ML model uses numeric arrays.

    The dashboard table uses formatted strings.
    """

    rows = []

    number_to_show = min(
        number_of_examples,
        len(X),
    )

    for i in range(number_to_show):

        rows.append(
            {
                "Example": i + 1,

                "X (Past)": format_values(
                    X[i]
                ),

                "y (Future)": format_values(
                    y[i]
                ),
            }
        )

    return rows


# =========================================================
# 5. TRAIN MULTI-STEP MODEL
# =========================================================

def train_forecasting_model(
    df,
    input_window=6,
    forecast_horizon=3,
):
    """
    Train one Random Forest model for each
    future forecast step.

    Model 1 -> t+1
    Model 2 -> t+2
    Model 3 -> t+3
    """

    # Create numeric supervised-learning dataset
    X, y, origins = create_multistep_dataset(
        df,
        input_window,
        forecast_horizon,
    )

    models = []

    # Store predictions made on training examples
    predictions = np.zeros_like(y)

    # Train one model for each future step
    for step in range(forecast_horizon):

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )

        # Example:
        #
        # X -> [past 6 observations]
        #
        # y[:, 0] -> first future value
        # y[:, 1] -> second future value
        # y[:, 2] -> third future value

        model.fit(
            X,
            y[:, step],
        )

        # Prediction on the training examples
        predictions[:, step] = model.predict(X)

        models.append(model)

    return (
        models,
        X,
        y,
        predictions,
        origins,
    )


# =========================================================
# 6. FORECAST FROM LATEST OBSERVATIONS
# =========================================================

def make_future_forecast(
    df,
    models,
    input_window,
):
    """
    Use the latest known observations
    to forecast future values.
    """

    # Take the most recent observations
    latest_window = (
        df["Value"]
        .to_numpy()[-input_window:]
    )

    # Random Forest expects 2D input
    X_future = latest_window.reshape(
        1,
        -1,
    )

    forecast = []

    # Each model predicts one future step
    for model in models:

        prediction = model.predict(
            X_future
        )[0]

        forecast.append(
            prediction
        )

    return np.array(forecast)


# =========================================================
# 7. PAGE LAYOUT
# =========================================================

layout = dbc.Container(
    [

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        html.H2(
            "Forecasting with Machine Learning",
            className="mt-4 mb-2",
        ),

        html.P(
            "See how a time series can be transformed "
            "into a supervised machine-learning problem.",
            className="text-muted mb-4",
        ),


        # -------------------------------------------------
        # CONTROLS
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "Forecast Controls",
                        className="mb-3",
                    ),

                    # Input window
                    html.Label(
                        "Past observations used as input",
                        className="fw-bold",
                    ),

                    dcc.Slider(
                        id="forecast-input-window",
                        min=3,
                        max=12,
                        step=1,
                        value=6,

                        marks={
                            i: str(i)
                            for i in range(
                                3,
                                13,
                            )
                        },

                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                    ),

                    html.Br(),

                    # Forecast horizon
                    html.Label(
                        "Number of future steps",
                        className="fw-bold",
                    ),

                    dcc.Slider(
                        id="forecast-horizon",
                        min=1,
                        max=6,
                        step=1,
                        value=3,

                        marks={
                            i: str(i)
                            for i in range(
                                1,
                                7,
                            )
                        },

                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                    ),

                ]
            ),
            className="mb-4",
        ),


        # -------------------------------------------------
        # MAIN VISUAL EXPLANATION
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "1. Turn the Time Series into an ML Problem",
                        className="mb-3",
                    ),

                    dcc.Graph(
                        id="forecast-window-chart",
                    ),

                    html.Div(
                        id="forecast-window-explanation",
                    ),

                ]
            ),
            className="mb-4",
        ),


        # -------------------------------------------------
        # SUPERVISED LEARNING TABLE
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "2. From Time Series to Training Examples",
                        className="mb-3",
                    ),

                    html.P(
                        [
                            "Each row represents one training example: ",
                            html.Strong(
                                "past observations → future observations"
                            ),
                        ],
                        className="text-muted",
                    ),

                    html.Div(
                        id="forecast-training-table",
                    ),

                ]
            ),
            className="mb-4",
        ),


        # -------------------------------------------------
        # FORECAST CHART
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "3. Machine Learning Forecast",
                        className="mb-3",
                    ),

                    dcc.Graph(
                        id="forecast-result-chart",
                    ),

                    html.Div(
                        id="forecast-result-explanation",
                    ),

                ]
            ),
            className="mb-4",
        ),


        # -------------------------------------------------
        # KEY TAKEAWAY
        # -------------------------------------------------

        dbc.Alert(
            [

                html.H5(
                    "Key Takeaway",
                ),

                html.P(
                    "Machine learning does not receive the "
                    "time series as one long sequence."
                ),

                html.P(
                    "We create features from past observations "
                    "and use future observations as targets."
                ),

                html.P(
                    [
                        html.Strong(
                            "Past → X → Machine Learning Model → Future"
                        )
                    ],
                    className="mb-0",
                ),

            ],
            color="info",
            className="mb-4",
        ),

    ],
    fluid=True,
)


# =========================================================
# 8. CALLBACK
# =========================================================

@dash.callback(
    Output(
        "forecast-window-chart",
        "figure",
    ),

    Output(
        "forecast-window-explanation",
        "children",
    ),

    Output(
        "forecast-training-table",
        "children",
    ),

    Output(
        "forecast-result-chart",
        "figure",
    ),

    Output(
        "forecast-result-explanation",
        "children",
    ),

    Input(
        "forecast-input-window",
        "value",
    ),

    Input(
        "forecast-horizon",
        "value",
    ),
)
def update_forecasting_dashboard(
    input_window,
    forecast_horizon,
):

    # =====================================================
    # Generate data
    # =====================================================

    df = generate_data()

    values = df["Value"].to_numpy()


    # =====================================================
    # Choose a demonstration point
    # =====================================================

    origin = 55

    past_start = (
        origin - input_window
    )

    future_end = min(
        origin + forecast_horizon,
        len(df),
    )


    # =====================================================
    # CHART 1
    # Show how the time series becomes X and y
    # =====================================================

    fig_window = go.Figure()


    # -----------------------------------------------------
    # Entire time series
    # -----------------------------------------------------

    fig_window.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Time series",

            line=dict(
                width=2,
            ),
        )
    )


    # -----------------------------------------------------
    # Past window = X
    # -----------------------------------------------------

    fig_window.add_trace(
        go.Scatter(
            x=df["Time"].iloc[
                past_start:origin
            ],

            y=df["Value"].iloc[
                past_start:origin
            ],

            mode="lines+markers",

            name="Past / X",

            line=dict(
                width=4,
            ),
        )
    )


    # -----------------------------------------------------
    # Future window = y
    # -----------------------------------------------------

    fig_window.add_trace(
        go.Scatter(
            x=df["Time"].iloc[
                origin:future_end
            ],

            y=df["Value"].iloc[
                origin:future_end
            ],

            mode="lines+markers",

            name="Future / y",

            line=dict(
                width=4,
            ),
        )
    )


    # -----------------------------------------------------
    # Forecast origin
    # -----------------------------------------------------

    fig_window.add_vline(
        x=origin,
        line_dash="dash",
    )


    fig_window.update_layout(
        title=(
            "Past observations become X, "
            "future observations become y"
        ),

        xaxis_title="Time",
        yaxis_title="Value",

        template="plotly_white",

        hovermode="x unified",
    )


    # =====================================================
    # EXPLANATION
    # =====================================================

    past_values = np.round(
        values[
            past_start:origin
        ],
        1,
    )

    future_values = np.round(
        values[
            origin:future_end
        ],
        1,
    )


    explanation = html.Div(
        [

            html.P(
                [
                    html.Strong(
                        "Past / X: "
                    ),

                    "The model can see these observations.",
                ]
            ),

            html.P(
                [
                    html.Strong(
                        "Future / y: "
                    ),

                    "These are the values we want "
                    "the model to predict.",
                ]
            ),

            html.P(
                [
                    html.Strong(
                        "Forecast origin: "
                    ),

                    "The point where known data ends "
                    "and the future begins.",
                ]
            ),

            html.Div(
                [
                    html.Strong("X = "),

                    html.Code(
                        format_values(
                            past_values
                        )
                    ),
                ],
                className="mb-2",
            ),

            html.Div(
                [
                    html.Strong("y = "),

                    html.Code(
                        format_values(
                            future_values
                        )
                    ),
                ],

            ),

        ]
    )


    # =====================================================
    # CREATE SUPERVISED-LEARNING DATASET
    # =====================================================

    X, y, origins = create_multistep_dataset(
        df,
        input_window,
        forecast_horizon,
    )


    # =====================================================
    # CREATE EDUCATIONAL TABLE
    # =====================================================

    rows = create_training_table_data(
        X,
        y,
        number_of_examples=5,
    )

    training_df = pd.DataFrame(
        rows
    )


    # -----------------------------------------------------
    # Create Bootstrap table
    # -----------------------------------------------------

    table = dbc.Table.from_dataframe(
        training_df,

        striped=True,
        bordered=True,
        hover=True,
        responsive=True,

        class_name="align-middle",

        style={
            "textAlign": "center",
        },
    )


    # =====================================================
    # TRAIN MODEL
    # =====================================================

    (
        models,
        X_all,
        y_all,
        predictions,
        training_origins,
    ) = train_forecasting_model(
        df,
        input_window,
        forecast_horizon,
    )


    # =====================================================
    # FUTURE FORECAST
    # =====================================================

    future_forecast = make_future_forecast(
        df,
        models,
        input_window,
    )


    last_time = df["Time"].iloc[-1]

    future_times = np.arange(
        last_time + 1,
        last_time + 1 + forecast_horizon,
    )


    # =====================================================
    # CHART 2
    # =====================================================

    fig_forecast = go.Figure()


    # -----------------------------------------------------
    # Historical data
    # -----------------------------------------------------

    fig_forecast.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Known history",
        )
    )


    # -----------------------------------------------------
    # Latest input window
    # -----------------------------------------------------

    fig_forecast.add_trace(
        go.Scatter(
            x=df["Time"].iloc[
                -input_window:
            ],

            y=df["Value"].iloc[
                -input_window:
            ],

            mode="lines+markers",

            name="Input to model",

            line=dict(
                width=4,
            ),
        )
    )


    # -----------------------------------------------------
    # Future forecast
    # -----------------------------------------------------

    fig_forecast.add_trace(
        go.Scatter(
            x=future_times,
            y=future_forecast,

            mode="lines+markers",

            name="ML Forecast",

            line=dict(
                dash="dash",
                width=3,
            ),
        )
    )


    # -----------------------------------------------------
    # Forecast origin
    # -----------------------------------------------------

    fig_forecast.add_vline(
        x=last_time,
        line_dash="dash",
    )


    fig_forecast.update_layout(
        title=(
            "Using the latest past window "
            "to predict the future"
        ),

        xaxis_title="Time",
        yaxis_title="Value",

        template="plotly_white",

        hovermode="x unified",
    )


    # =====================================================
    # FINAL EXPLANATION
    # =====================================================

    mae = mean_absolute_error(
        y_all.ravel(),
        predictions.ravel(),
    )


    result_explanation = html.Div(
        [

            html.P(
                [
                    html.Strong(
                        "Training: "
                    ),

                    "The model sees many historical examples "
                    "of past values followed by future values.",
                ]
            ),

            html.P(
                [
                    html.Strong(
                        "Forecasting: "
                    ),

                    "Now we give the model only the latest "
                    f"{input_window} observations.",
                ]
            ),

            html.P(
                [
                    html.Strong(
                        "Prediction: "
                    ),

                    "The model produces the next "
                    f"{forecast_horizon} time steps.",
                ]
            ),

            html.P(
                [
                    html.Strong(
                        "Important idea: "
                    ),

                    "The future values are hidden when "
                    "we make the forecast.",
                ]
            ),

            html.P(
                [
                    "Illustrative training MAE: ",

                    html.Strong(
                        f"{mae:.2f}"
                    ),
                ],

                className="mb-0",
            ),

        ]
    )


    # =====================================================
    # RETURN ALL DASHBOARD OUTPUTS
    # =====================================================

    return (
        fig_window,
        explanation,
        table,
        fig_forecast,
        result_explanation,
    )