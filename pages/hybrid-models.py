import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/hybrid-models",
    name="Hybrid Models",
)


# =========================================================
# 1. DATA GENERATION
# =========================================================

def generate_data(
    n=150,
    noise_level=2.0,
    seasonal_strength=10.0,
    seed=42
):
    """
    Generate a simple time series containing:

    - Trend
    - Seasonality
    - Noise
    """

    rng = np.random.default_rng(seed)

    t = np.arange(n)

    trend = 0.15 * t

    seasonal = (
        seasonal_strength
        * np.sin(2 * np.pi * t / 12)
    )

    noise = rng.normal(
        0,
        noise_level,
        n
    )

    y = (
        20
        + trend
        + seasonal
        + noise
    )

    return pd.DataFrame({
        "Time": t,
        "Value": y
    })


# =========================================================
# 2. FEATURE CREATION
# =========================================================

def create_features(df):
    """
    Create lag features for the residual model.
    """

    X = pd.DataFrame(index=df.index)

    X["Time"] = df["Time"]

    X["Lag_1"] = (
        df["Value"]
        .shift(1)
    )

    X["Lag_2"] = (
        df["Value"]
        .shift(2)
    )

    X["Lag_12"] = (
        df["Value"]
        .shift(12)
    )

    return X


# =========================================================
# 3. FIRST MODEL
# =========================================================

def fit_first_model(df):
    """
    Model 1 learns the main trend.

    Input:
        Time

    Target:
        Value
    """

    X = df[["Time"]]

    y = df["Value"]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    y_fit = model.predict(X)

    return model, y_fit


# =========================================================
# 4. HYBRID MODEL
# =========================================================

def fit_hybrid_model(df):
    """
    Two-stage hybrid model:

    Model 1:
        Learns the main trend.

    Residual:
        Actual - Model 1 prediction.

    Model 2:
        Learns the residual pattern.

    Hybrid:
        Model 1 prediction + Model 2 prediction.
    """

    # -----------------------------------------------------
    # Model 1
    # -----------------------------------------------------

    model_1, y_fit = fit_first_model(df)

    # -----------------------------------------------------
    # Calculate residuals
    # -----------------------------------------------------

    residuals = (
        df["Value"].values
        - y_fit
    )

    # -----------------------------------------------------
    # Create features for Model 2
    # -----------------------------------------------------

    X_residual = create_features(df)

    valid = X_residual.notna().all(axis=1)

    X_residual_valid = (
        X_residual.loc[valid]
    )

    residuals_valid = (
        residuals[valid]
    )

    # -----------------------------------------------------
    # Model 2
    # -----------------------------------------------------

    model_2 = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model_2.fit(
        X_residual_valid,
        residuals_valid
    )

    # -----------------------------------------------------
    # Predict residuals
    # -----------------------------------------------------

    residual_prediction = np.zeros(
        len(df)
    )

    residual_prediction[valid] = (
        model_2.predict(
            X_residual_valid
        )
    )

    # -----------------------------------------------------
    # Final hybrid prediction
    # -----------------------------------------------------

    hybrid_prediction = (
        y_fit
        + residual_prediction
    )

    return (
        y_fit,
        residuals,
        residual_prediction,
        hybrid_prediction
    )


# =========================================================
# 5. PAGE LAYOUT
# =========================================================

layout = dbc.Container(
    [

        # -------------------------------------------------
        # PAGE TITLE
        # -------------------------------------------------

        html.H2(
            "Hybrid Models",
            className="mt-4 mb-2"
        ),

        html.P(
            "See how a second model can learn what the first "
            "model could not explain.",
            className="text-muted mb-4"
        ),

        # -------------------------------------------------
        # CONTROLS
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "Experiment Controls",
                        className="mb-3"
                    ),

                    # Seasonal strength
                    html.Label(
                        "Seasonal Strength",
                        className="fw-bold"
                    ),

                    dcc.Slider(
                        id="hybrid-seasonal-strength",
                        min=0,
                        max=20,
                        step=1,
                        value=10,
                        marks={
                            0: "0",
                            5: "5",
                            10: "10",
                            15: "15",
                            20: "20"
                        },
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True
                        }
                    ),

                    html.Br(),

                    # Noise
                    html.Label(
                        "Noise Level",
                        className="fw-bold"
                    ),

                    dcc.Slider(
                        id="hybrid-noise",
                        min=0,
                        max=5,
                        step=0.5,
                        value=2,
                        marks={
                            0: "0",
                            1: "1",
                            2: "2",
                            3: "3",
                            4: "4",
                            5: "5"
                        },
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True
                        }
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # CONCEPT EXPLANATION
        # -------------------------------------------------

        dbc.Alert(
            [
                html.H5(
                    "How the Hybrid Model Works",
                    className="mb-3"
                ),

                html.P(
                    "Model 1 learns the main pattern of the "
                    "time series."
                ),

                html.P(
                    "The residuals show what Model 1 missed."
                ),

                html.P(
                    "Model 2 learns these residual patterns."
                ),

                html.P(
                    [
                        html.Strong("Hybrid prediction = "),
                        "Model 1 prediction + Model 2 residual prediction"
                    ],
                    className="mb-0"
                )
            ],
            color="info",
            className="mb-4"
        ),

        # -------------------------------------------------
        # CHART 1
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "1. Original Time Series",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="hybrid-original-chart"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # CHART 2
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "2. Model 1: Main Pattern",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="hybrid-model1-chart"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # CHART 3
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "3. Residuals: What Model 1 Missed",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="hybrid-residual-chart"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # CHART 4
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "4. Hybrid Model",
                        className="mb-3"
                    ),

                    dcc.Graph(
                        id="hybrid-final-chart"
                    )

                ]
            ),
            className="mb-4"
        ),

        # -------------------------------------------------
        # FINAL EXPLANATION
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [

                    html.H5(
                        "What Should You Notice?",
                        className="mb-3"
                    ),

                    html.Div(
                        id="hybrid-explanation"
                    )

                ]
            ),
            className="mb-4"
        )

    ],
    fluid=True
)


# =========================================================
# 6. CALLBACK
# =========================================================

@dash.callback(
    Output(
        "hybrid-original-chart",
        "figure"
    ),

    Output(
        "hybrid-model1-chart",
        "figure"
    ),

    Output(
        "hybrid-residual-chart",
        "figure"
    ),

    Output(
        "hybrid-final-chart",
        "figure"
    ),

    Output(
        "hybrid-explanation",
        "children"
    ),

    Input(
        "hybrid-seasonal-strength",
        "value"
    ),

    Input(
        "hybrid-noise",
        "value"
    )
)
def update_hybrid_dashboard(
    seasonal_strength,
    noise_level
):

    # -----------------------------------------------------
    # Generate data
    # -----------------------------------------------------

    df = generate_data(
        seasonal_strength=seasonal_strength,
        noise_level=noise_level
    )

    # -----------------------------------------------------
    # Fit hybrid model
    # -----------------------------------------------------

    (
        y_fit,
        residuals,
        residual_prediction,
        hybrid_prediction
    ) = fit_hybrid_model(df)

    # =====================================================
    # CHART 1
    # =====================================================

    fig_original = go.Figure()

    fig_original.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Actual"
        )
    )

    fig_original.update_layout(
        title="Original Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified"
    )

    # =====================================================
    # CHART 2
    # =====================================================

    fig_model1 = go.Figure()

    fig_model1.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Actual"
        )
    )

    fig_model1.add_trace(
        go.Scatter(
            x=df["Time"],
            y=y_fit,
            mode="lines",
            name="Model 1"
        )
    )

    fig_model1.update_layout(
        title="Model 1: Learning the Main Trend",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified"
    )

    # =====================================================
    # CHART 3
    # =====================================================

    fig_residual = go.Figure()

    fig_residual.add_trace(
        go.Scatter(
            x=df["Time"],
            y=residuals,
            mode="lines",
            name="Residual"
        )
    )

    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_residual.update_layout(
        title="Residuals: What Model 1 Missed",
        xaxis_title="Time",
        yaxis_title="Residual",
        template="plotly_white",
        hovermode="x unified"
    )

    # =====================================================
    # CHART 4
    # =====================================================

    fig_final = go.Figure()

    fig_final.add_trace(
        go.Scatter(
            x=df["Time"],
            y=df["Value"],
            mode="lines",
            name="Actual"
        )
    )

    fig_final.add_trace(
        go.Scatter(
            x=df["Time"],
            y=y_fit,
            mode="lines",
            name="Model 1"
        )
    )

    fig_final.add_trace(
        go.Scatter(
            x=df["Time"],
            y=hybrid_prediction,
            mode="lines",
            name="Hybrid Model"
        )
    )

    fig_final.update_layout(
        title="Model 1 vs. Hybrid Model",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified"
    )

    # =====================================================
    # EXPLANATION
    # =====================================================

    explanation = html.Div(
        [

            html.P(
                [
                    html.Strong("Step 1 — Model 1: "),
                    "The first model uses Time to learn the "
                    "main trend."
                ]
            ),

            html.P(
                [
                    html.Strong("Step 2 — Residuals: "),
                    "We subtract Model 1 predictions from the "
                    "actual values."
                ]
            ),

            html.P(
                [
                    html.Strong("Step 3 — Model 2: "),
                    "The second model uses lag features to "
                    "learn patterns remaining in the residuals."
                ]
            ),

            html.P(
                [
                    html.Strong("Step 4 — Hybrid: "),
                    "The final prediction combines both models."
                ]
            ),

            html.Code(
                "Hybrid = Model 1 + Model 2(residuals)"
            )

        ]
    )

    return (
        fig_original,
        fig_model1,
        fig_residual,
        fig_final,
        explanation
    )
