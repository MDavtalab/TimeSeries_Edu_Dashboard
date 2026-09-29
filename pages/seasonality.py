"""
=========================================================
Time Series Learning Lab (TSLL)
Seasonality
=========================================================

This module demonstrates:

    - Seasonal patterns
    - Fourier features
    - Annual, six-month, quarterly, and monthly cycles
    - Fourier order
    - Seasonal features versus lag features
    - Periodogram analysis
    - Forecasting with Fourier features
"""

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, dash_table
from sklearn.linear_model import LinearRegression


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/seasonality",
    name="Seasonality",
)


# =========================================================
# Configuration
# =========================================================

FORECAST_DAYS = 90

# The seasonal cycles are expressed in days because
# our example uses daily observations.

SEASONAL_CYCLES = {
    "Annual (1 year)": 365.25,
    "Six-month": 365.25 / 2,
    "Quarterly (3 months)": 365.25 / 4,
    "Monthly (approximately)": 365.25 / 12,
}

CYCLE_OPTIONS = [
    {
        "label": name,
        "value": name,
    }
    for name in SEASONAL_CYCLES
]


# =========================================================
# Generate Predefined Dataset
# =========================================================

def generate_seasonal_data():
    """
    Generate approximately two years of daily sales data.

    The data contains:
        - Linear trend
        - Annual seasonality
        - Quarterly seasonality
        - Monthly seasonality
        - Random noise
    """

    np.random.seed(42)

    n_days = 730

    dates = pd.date_range(
        start="2023-01-01",
        periods=n_days,
        freq="D",
    )

    t = np.arange(n_days)

    # Long-term upward trend.

    trend = 100 + 0.04 * t

    # Annual seasonal component.

    annual = (
        25
        * np.sin(
            2 * np.pi * t / 365.25
        )
    )

    # Quarterly seasonal component.

    quarterly = (
        12
        * np.sin(
            2 * np.pi * t / (365.25 / 4)
        )
    )

    # Monthly seasonal component.

    monthly = (
        8
        * np.sin(
            2 * np.pi * t / (365.25 / 12)
        )
    )

    # Random fluctuations.

    noise = np.random.normal(
        loc=0,
        scale=6,
        size=n_days,
    )

    sales = (
        trend
        + annual
        + quarterly
        + monthly
        + noise
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Time": t,
            "Sales": sales,
        }
    )


sales_data = generate_seasonal_data()


# =========================================================
# Fourier Feature Generation
# =========================================================

def create_fourier_features(
    time_values,
    period,
    order,
):
    """
    Generate Fourier features for a given seasonal period.

    For each harmonic k:

        sin(2 * pi * k * t / period)
        cos(2 * pi * k * t / period)

    Parameters
    ----------
    time_values : array-like
        Numeric time index.

    period : float
        Seasonal period in time steps.

    order : int
        Number of Fourier harmonics.

    Returns
    -------
    pandas.DataFrame
        Fourier feature columns.
    """

    time_values = np.asarray(time_values)

    features = {}

    for k in range(1, order + 1):

        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * time_values / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * time_values / period
        )

    return pd.DataFrame(features)


def create_selected_fourier_features(
    time_values,
    selected_cycles,
    order,
):
    """
    Generate Fourier features for all selected cycles.
    """

    feature_frames = []

    for cycle_name in selected_cycles:

        period = SEASONAL_CYCLES[cycle_name]

        cycle_features = create_fourier_features(
            time_values=time_values,
            period=period,
            order=order,
        )

        # Rename columns to identify their seasonal cycle.

        cycle_features = cycle_features.rename(
            columns={
                column: f"{cycle_name}_{column}"
                for column in cycle_features.columns
            }
        )

        feature_frames.append(cycle_features)

    if not feature_frames:

        return pd.DataFrame(
            index=np.arange(len(time_values))
        )

    return pd.concat(
        feature_frames,
        axis=1,
    )


# =========================================================
# Fourier Model
# =========================================================

def fit_fourier_model(
    data,
    selected_cycles,
    order,
):
    """
    Fit a regression model using:

        - A linear time trend
        - Selected Fourier features
    """

    time_values = data["Time"].to_numpy()

    fourier_features = create_selected_fourier_features(
        time_values=time_values,
        selected_cycles=selected_cycles,
        order=order,
    )

    # Add a linear trend feature.

    X = pd.DataFrame(
        {
            "Time": time_values,
        }
    )

    X = pd.concat(
        [
            X.reset_index(drop=True),
            fourier_features.reset_index(drop=True),
        ],
        axis=1,
    )

    model = LinearRegression()

    model.fit(
        X,
        data["Sales"],
    )

    return model


def predict_fourier_model(
    model,
    time_values,
    selected_cycles,
    order,
):
    """
    Generate predictions using the fitted Fourier model.
    """

    time_values = np.asarray(time_values)

    fourier_features = create_selected_fourier_features(
        time_values=time_values,
        selected_cycles=selected_cycles,
        order=order,
    )

    X = pd.DataFrame(
        {
            "Time": time_values,
        }
    )

    X = pd.concat(
        [
            X.reset_index(drop=True),
            fourier_features.reset_index(drop=True),
        ],
        axis=1,
    )

    return model.predict(X)


# =========================================================
# Main Seasonal Chart
# =========================================================

def create_seasonality_figure(
    selected_cycles,
    order,
    show_forecast,
):
    """
    Create a chart with observed sales, fitted values,
    and an optional 90-day forecast.
    """

    data = sales_data.copy()

    # Fit the model on the historical data.

    model = fit_fourier_model(
        data=data,
        selected_cycles=selected_cycles,
        order=order,
    )

    # Historical fitted values.

    historical_predictions = predict_fourier_model(
        model=model,
        time_values=data["Time"].to_numpy(),
        selected_cycles=selected_cycles,
        order=order,
    )

    fig = go.Figure()

    # -----------------------------------------------------
    # Original Data
    # -----------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Sales"],
            mode="lines",
            name="Observed Sales",
            line=dict(
                color="#34495E",
                width=1.5,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Sales: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    # -----------------------------------------------------
    # Fitted Seasonal Model
    # -----------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=historical_predictions,
            mode="lines",
            name="Fitted Trend + Seasonality",
            line=dict(
                color="#E67E22",
                width=3,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Fitted Value: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    # -----------------------------------------------------
    # Forecast
    # -----------------------------------------------------

    if show_forecast:

        future_time = np.arange(
            len(data),
            len(data) + FORECAST_DAYS,
        )

        future_dates = pd.date_range(
            start=data["Date"].iloc[-1]
            + pd.Timedelta(days=1),

            periods=FORECAST_DAYS,

            freq="D",
        )

        future_predictions = predict_fourier_model(
            model=model,
            time_values=future_time,
            selected_cycles=selected_cycles,
            order=order,
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode="lines",
                name="90-Day Forecast",
                line=dict(
                    color="#27AE60",
                    width=3,
                    dash="dash",
                ),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Forecast: %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

        # Mark the forecast boundary.

        fig.add_vline(
            x=data["Date"].iloc[-1].timestamp() * 1000,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top",
        )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.update_layout(
        title="Seasonality and Fourier-Based Forecasting",

        xaxis_title="Date",

        yaxis_title="Sales",

        template="plotly_white",

        hovermode="x unified",

        height=520,

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
            t=100,
            b=50,
        ),
    )

    return fig


# =========================================================
# Periodogram
# =========================================================

def calculate_periodogram(data):
    """
    Calculate a simple periodogram using the Fast Fourier
    Transform (FFT).

    The input series is centered by subtracting its mean.

    Returns
    -------
    periods : numpy.ndarray
        Periods in days.

    power : numpy.ndarray
        Power at each frequency.
    """

    values = data["Sales"].to_numpy()

    # Remove the mean to reduce the zero-frequency component.

    values = values - np.mean(values)

    n = len(values)

    # Calculate the real FFT.

    fft_values = np.fft.rfft(values)

    # Calculate the corresponding frequencies.

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0,
    )

    # Calculate the power spectrum.

    power = (
        np.abs(fft_values) ** 2
    ) / n

    # Remove the zero frequency.

    frequencies = frequencies[1:]

    power = power[1:]

    # Convert frequency to period.

    periods = 1 / frequencies

    return periods, power


def create_periodogram_figure():
    """
    Create a periodogram chart showing power against period.
    """

    periods, power = calculate_periodogram(
        sales_data
    )

    # Keep periods within a useful range for this example.

    mask = (
        (periods >= 10)
        & (periods <= 500)
    )

    periods = periods[mask]

    power = power[mask]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=periods,
            y=power,
            mode="lines",
            name="Periodogram",
            line=dict(
                color="#2980B9",
                width=2,
            ),
            hovertemplate=(
                "Period: %{x:.1f} days<br>"
                "Power: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    # Highlight the predefined seasonal periods.

    for cycle_name, period in SEASONAL_CYCLES.items():

        fig.add_vline(
            x=period,
            line_dash="dash",
            line_color="gray",
            annotation_text=cycle_name,
            annotation_position="top",
        )

    fig.update_layout(
        title="Periodogram: Identifying Seasonal Periods",

        xaxis_title="Period (days)",

        yaxis_title="Power",

        template="plotly_white",

        height=450,

        margin=dict(
            l=50,
            r=30,
            t=100,
            b=50,
        ),
    )

    fig.update_xaxes(
        type="log",
        title="Period (days, logarithmic scale)",
    )

    return fig


# =========================================================
# Seasonal Features vs. Lag Features
# =========================================================

def create_feature_comparison():
    """
    Create a small dataset that compares:

        - A seasonal calendar feature
        - A lag feature
    """

    dates = pd.date_range(
        start="2024-01-01",
        periods=400,
        freq="D",
    )

    time_values = np.arange(len(dates))

    # A simple repeating annual seasonal pattern.

    seasonal_values = np.sin(
        2 * np.pi * time_values / 365.25
    )

    # Generate a simple series with trend and seasonality.

    values = (
        100
        + 0.05 * time_values
        + 20 * seasonal_values
    )

    df = pd.DataFrame(
        {
            "Date": dates,
            "Sales": values,
        }
    )

    # Calendar-based features.

    df["Month"] = df["Date"].dt.month

    # Fourier seasonal feature.

    df["Annual_Sin"] = np.sin(
        2 * np.pi * time_values / 365.25
    )

    df["Annual_Cos"] = np.cos(
        2 * np.pi * time_values / 365.25
    )

    # Lag feature: the value from 30 days earlier.

    df["Lag_30"] = df["Sales"].shift(30)

    # Select representative dates for display.

    selected_rows = [
        0, 30, 59, 90, 120,
        181, 273, 364, 394,
    ]

    comparison = df.iloc[
        selected_rows
    ].copy()

    return comparison


def create_feature_comparison_table():
    """
    Create a Dash DataTable showing seasonal and lag features.
    """

    comparison = create_feature_comparison()

    display_data = comparison[
        [
            "Date",
            "Sales",
            "Month",
            "Annual_Sin",
            "Annual_Cos",
            "Lag_30",
        ]
    ].copy()

    display_data["Date"] = (
        display_data["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_data = (
        display_data
        .round(2)
        .astype(object)
        .where(pd.notna(display_data), None)
    )

    return display_data.to_dict("records")


# =========================================================
# Python Code Generator
# =========================================================

def create_seasonality_code(
    selected_cycles,
    order,
):
    """
    Generate Python code demonstrating Fourier features.
    """

    if not selected_cycles:

        return (
            "# Select at least one seasonal cycle "
            "to generate Fourier features."
        )

    lines = [
        "import numpy as np",
        "import pandas as pd",
        "",
        "# Create a numeric time index",
        "df['Time'] = np.arange(len(df))",
        "",
        "# Generate Fourier features",
    ]

    for cycle_name in selected_cycles:

        period = SEASONAL_CYCLES[cycle_name]

        safe_name = (
            cycle_name
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        lines.append(
            f"# {cycle_name}: period = {period:.2f} days"
        )

        for k in range(1, order + 1):

            lines.append(
                f"df['{safe_name}_sin_{k}'] = "
                f"np.sin(2 * np.pi * {k} * df['Time'] / {period:.4f})"
            )

            lines.append(
                f"df['{safe_name}_cos_{k}'] = "
                f"np.cos(2 * np.pi * {k} * df['Time'] / {period:.4f})"
            )

        lines.append("")

    lines.extend(
        [
            "# Fit a regression model using the features",
            "from sklearn.linear_model import LinearRegression",
            "",
            "# Select your Fourier columns and fit the model.",
        ]
    )

    return "\n".join(lines)


# =========================================================
# Layout
# =========================================================

layout = dbc.Container(
    [
        # -------------------------------------------------
        # Page Title
        # -------------------------------------------------

        html.H2(
            "🔄 Seasonality",
            className="page-title",
        ),

        html.Hr(),

        # -------------------------------------------------
        # Introduction
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("What Is Seasonality?"),

                    html.P(
                        """
                        Seasonality is a repeating pattern in a
                        time series that occurs at regular intervals.
                        For example, sales may increase every
                        December, or electricity demand may follow
                        a daily pattern.
                        """
                    ),

                    html.P(
                        """
                        Unlike a trend, which describes the
                        long-term direction of a series, seasonality
                        describes recurring changes over a known
                        period.
                        """
                    ),

                    html.Div(
                        "Observed Data = Trend + Seasonality + Noise",
                        className="formula",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Fourier Features
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Fourier Features",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Fourier features represent repeating
                        patterns using sine and cosine functions.
                        They are useful when a time series has
                        a seasonal cycle that repeats over a
                        known period.
                        """
                    ),

                    html.Div(
                        "sin(2πkt / m)    and    cos(2πkt / m)",
                        className="formula",
                    ),

                    html.P(
                        """
                        Here, t is the time index, m is the
                        seasonal period, and k is the harmonic
                        number. Each harmonic adds another pair
                        of sine and cosine features.
                        """
                    ),

                    html.P(
                        """
                        The first harmonic captures the broad
                        seasonal shape. Higher harmonics allow
                        the model to represent more complex
                        seasonal patterns.
                        """
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Controls
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Explore Seasonal Cycles",
                        className="section-title",
                    ),

                    html.Label(
                        "Select Seasonal Cycles",
                        className="fw-bold",
                    ),

                    dcc.Dropdown(
                        id="seasonality-cycle-selector",

                        options=CYCLE_OPTIONS,

                        value=[
                            "Annual (1 year)",
                            "Quarterly (3 months)",
                        ],

                        multi=True,

                        placeholder="Select seasonal cycles...",
                    ),

                    html.Br(),

                    html.Label(
                        "Fourier Order (Number of Harmonics)",
                        className="fw-bold",
                    ),

                    dcc.Slider(
                        id="seasonality-order",

                        min=1,
                        max=8,
                        step=1,
                        value=3,

                        marks={
                            1: "1",
                            2: "2",
                            4: "4",
                            6: "6",
                            8: "8",
                        },

                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                    ),

                    html.Br(),

                    dbc.Checklist(
                        id="seasonality-forecast-toggle",

                        options=[
                            {
                                "label": "Show 90-day forecast",
                                "value": "forecast",
                            }
                        ],

                        value=["forecast"],

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
                        "Seasonal Pattern and Forecast",
                        className="section-title",
                    ),

                    html.P(
                        """
                        The orange line shows the fitted trend
                        and seasonal pattern. The dashed green
                        line shows the forecast for the next
                        90 days.
                        """
                    ),

                    dcc.Graph(
                        id="seasonality-main-graph",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Fourier Explanation
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Understanding Fourier Features",
                        className="section-title",
                    ),

                    html.Div(
                        id="seasonality-explanation",
                    ),

                    html.Hr(),

                    html.H5("What Does Fourier Order Mean?"),

                    html.Ul(
                        [
                            html.Li(
                                "Order 1 uses one sine-cosine pair "
                                "to represent a smooth seasonal cycle."
                            ),

                            html.Li(
                                "Order 2 adds a second harmonic, "
                                "allowing more variation within "
                                "the seasonal cycle."
                            ),

                            html.Li(
                                "Higher orders allow more complex "
                                "seasonal shapes but can overfit "
                                "the training data."
                            ),
                        ]
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Seasonal vs. Lag Features
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Seasonal Features vs. Lag Features",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Seasonal features represent a recurring
                        position within a cycle. Lag features
                        represent values observed a fixed number
                        of time steps earlier.
                        """
                    ),

                    html.H5("Example: Monthly Seasonality"),

                    html.P(
                        """
                        In monthly data, a seasonal feature
                        can represent the month of the year.
                        January is January every year, and
                        December is December every year.
                        """
                    ),

                    html.P(
                        """
                        A Lag 12 feature, however, means the
                        observation from 12 time steps earlier.
                        For monthly data, this corresponds to
                        the same calendar month in the previous
                        year. For daily data, Lag 12 means
                        12 days earlier—not the same month
                        of the previous year.
                        """
                    ),

                    html.Div(
                        "Seasonal feature: recurring calendar position",
                        className="formula",
                    ),

                    html.Div(
                        "Lag feature: value at time t - k",
                        className="formula",
                    ),

                    html.Hr(),

                    html.H5(
                        "Daily Data: Calendar Month vs. Lag 30"
                    ),

                    html.P(
                        """
                        The table below uses daily observations.
                        Month is a calendar feature, while Lag 30
                        contains the sales value from 30 days earlier.
                        """
                    ),

                    dash_table.DataTable(
                        id="seasonality-comparison-table",

                        columns=[
                            {
                                "name": "Date",
                                "id": "Date",
                            },
                            {
                                "name": "Sales",
                                "id": "Sales",
                                "type": "numeric",
                            },
                            {
                                "name": "Month",
                                "id": "Month",
                                "type": "numeric",
                            },
                            {
                                "name": "Annual Sin",
                                "id": "Annual_Sin",
                                "type": "numeric",
                            },
                            {
                                "name": "Annual Cos",
                                "id": "Annual_Cos",
                                "type": "numeric",
                            },
                            {
                                "name": "Lag 30",
                                "id": "Lag_30",
                                "type": "numeric",
                            },
                        ],

                        data=create_feature_comparison_table(),

                        style_table={
                            "overflowX": "auto",
                        },

                        style_cell={
                            "textAlign": "center",
                            "padding": "8px",
                            "minWidth": "100px",
                        },

                        style_header={
                            "backgroundColor": "#16c4dd",
                            "fontWeight": "bold",
                        },

                        page_size=10,
                    ),

                    html.Br(),

                    dbc.Alert(
                        """
                        Key distinction: A calendar feature
                        identifies a recurring time position.
                        A lag feature carries information from
                        a previous observation. They can both
                        help forecasting, but they represent
                        different types of information.
                        """,
                        color="info",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Periodogram
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Periodogram: Discovering Seasonal Periods",
                        className="section-title",
                    ),

                    html.P(
                        """
                        A periodogram shows how much of a time
                        series' variation is associated with
                        different frequencies. Peaks can indicate
                        repeating patterns.
                        """
                    ),

                    html.P(
                        """
                        Frequency tells us how often a pattern
                        repeats per unit of time. Period is the
                        reciprocal of frequency.
                        """
                    ),

                    html.Div(
                        "Period = 1 / Frequency",
                        className="formula",
                    ),

                    html.P(
                        """
                        For daily data, a frequency of 1/365
                        cycles per day corresponds to a period
                        of approximately 365 days.
                        """
                    ),

                    dcc.Graph(
                        id="seasonality-periodogram",
                        figure=create_periodogram_figure(),
                    ),

                    dbc.Alert(
                        """
                        How to read the chart: Look for peaks
                        in power. A peak near 365 days may
                        indicate an annual pattern, while a
                        peak near 30 days may suggest a
                        monthly-like cycle. These are clues,
                        not proof of seasonality.
                        """,
                        color="info",
                    ),

                    html.P(
                        """
                        Important: The periodogram can also
                        show peaks caused by harmonics,
                        trends, noise, or other structures.
                        Use domain knowledge and validation
                        to confirm a useful seasonal period.
                        """
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
                        The following code updates when you
                        change the selected seasonal cycles
                        or Fourier order.
                        """
                    ),

                    html.Pre(
                        id="seasonality-code",
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
                                "Seasonality is a repeating pattern "
                                "that occurs at regular intervals."
                            ),

                            html.Li(
                                "Fourier features use sine and cosine "
                                "terms to represent seasonal cycles."
                            ),

                            html.Li(
                                "The seasonal period defines how "
                                "long one complete cycle lasts."
                            ),

                            html.Li(
                                "Fourier order controls how many "
                                "harmonics are used to represent "
                                "the seasonal shape."
                            ),

                            html.Li(
                                "Seasonal features represent "
                                "recurring time positions, while "
                                "lag features represent earlier "
                                "observations."
                            ),

                            html.Li(
                                "A periodogram helps identify "
                                "candidate seasonal periods by "
                                "showing peaks in frequency power."
                            ),

                            html.Li(
                                "A 90-day forecast is useful for "
                                "visualizing short-term predictions, "
                                "but forecast quality depends on "
                                "the model and data."
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
    Output("seasonality-main-graph", "figure"),
    Output("seasonality-explanation", "children"),
    Output("seasonality-code", "children"),

    Input("seasonality-cycle-selector", "value"),
    Input("seasonality-order", "value"),
    Input("seasonality-forecast-toggle", "value"),
)
def update_seasonality_dashboard(
    selected_cycles,
    order,
    forecast_toggle,
):

    selected_cycles = selected_cycles or []

    show_forecast = (
        "forecast" in (forecast_toggle or [])
    )

    # -----------------------------------------------------
    # Main Chart
    # -----------------------------------------------------

    figure = create_seasonality_figure(
        selected_cycles=selected_cycles,
        order=order,
        show_forecast=show_forecast,
    )

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------

    if selected_cycles:

        explanation_items = []

        for cycle_name in selected_cycles:

            period = SEASONAL_CYCLES[cycle_name]

            explanation_items.append(
                html.Li(
                    f"{cycle_name}: approximately "
                    f"{period:.1f} days per cycle."
                )
            )

        explanation = html.Div(
            [
                html.P(
                    f"You selected {len(selected_cycles)} "
                    f"seasonal cycle(s) with Fourier order {order}."
                ),

                html.Ul(explanation_items),

                html.P(
                    f"Each selected cycle generates "
                    f"{2 * order} Fourier features: "
                    f"{order} sine terms and {order} cosine terms."
                ),
            ],
            className="explanation",
        )

    else:

        explanation = dbc.Alert(
            """
            No seasonal cycles are selected. The model
            currently contains only a linear time trend.
            Select a cycle to add Fourier features.
            """,
            color="info",
        )

    # -----------------------------------------------------
    # Python Code
    # -----------------------------------------------------

    code = create_seasonality_code(
        selected_cycles=selected_cycles,
        order=order,
    )

    return figure, explanation, code