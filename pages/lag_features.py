"""
=========================================================
Time Series Learning Lab (TSLL)
Lag Features
=========================================================

This module demonstrates how lag features are created
using previous observations in a time series.

Students can:
    - Select multiple lag values.
    - View lagged values alongside the original data.
    - Understand how Pandas shift() works.
    - Learn how ACF and PACF help select lags.
"""

import dash
import dash_bootstrap_components as dbc
import pandas as pd

from dash import html, dcc, dash_table, Input, Output


# =========================================================
# Page Registration
# =========================================================

dash.register_page(
    __name__,
    path="/lag-features",
    name="Lag Features",
)


# =========================================================
# Sample Dataset
# =========================================================

# Monthly sales data for one year.

sales_data = pd.DataFrame(
    {
        "Month": [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec",
        ],
        "Sales": [
            100, 120, 115, 140,
            160, 150, 180, 200,
            190, 220, 240, 230,
        ],
    }
)

# Add a numeric index for easier calculations.

sales_data["Month_Index"] = range(len(sales_data))


# =========================================================
# Helper Functions
# =========================================================

def create_lagged_data(data, selected_lags):
    """
    Create lag features using pandas shift().

    A lag of k means that the value from k periods
    earlier is assigned to the current row.
    """

    result = data[["Month", "Sales"]].copy()

    for lag in sorted(selected_lags):
        result[f"Lag {lag}"] = data["Sales"].shift(lag)

    return result


def create_lag_explanation(selected_lags):
    """
    Generate an explanation for the selected lag values.
    """

    if not selected_lags:
        return html.P(
            "Select at least one lag to see how previous "
            "observations become new features.",
            className="explanation",
        )

    explanations = []

    for lag in sorted(selected_lags):

        if lag == 1:
            description = (
                "Lag 1 uses the sales value from the previous month."
            )

        elif lag == 2:
            description = (
                "Lag 2 uses the sales value from two months earlier."
            )

        elif lag == 12:
            description = (
                "Lag 12 uses the sales value from the same month "
                "in the previous year, when the data is monthly."
            )

        else:
            description = (
                f"Lag {lag} uses the sales value from "
                f"{lag} months earlier."
            )

        explanations.append(
            html.Li(description)
        )

    return html.Div(
        [
            html.H5("How to interpret the selected lags"),
            html.Ul(explanations),
        ],
        className="explanation",
    )


def create_python_code(selected_lags):
    """
    Generate Python code based on the selected lags.
    """

    if not selected_lags:
        return (
            "import pandas as pd\n\n"
            "# Select one or more lag values to generate the code."
        )

    lines = [
        "import pandas as pd",
        "",
        "# Create lag features using shift()",
        "df = sales_data.copy()",
        "",
    ]

    for lag in sorted(selected_lags):
        lines.append(
            f"df['Lag_{lag}'] = df['Sales'].shift({lag})"
        )

    lines.extend(
        [
            "",
            "# Display the original values and lag features",
            "print(df)",
        ]
    )

    return "\n".join(lines)


def create_lag_summary(selected_lags):
    """
    Explain the effect of the selected lag values.
    """

    if not selected_lags:
        return dbc.Alert(
            "Select one or more lags to see the summary.",
            color="info",
        )

    min_lag = min(selected_lags)
    max_lag = max(selected_lags)

    return dbc.Alert(
        [
            html.H5("Lag Summary"),
            html.P(
                f"You selected {len(selected_lags)} lag feature(s): "
                f"{', '.join(str(lag) for lag in sorted(selected_lags))}."
            ),
            html.P(
                f"The largest lag is {max_lag}. "
                f"Therefore, the first {max_lag} observations "
                "will have missing values in that lag column."
            ),
            html.P(
                f"The smallest lag is {min_lag}. "
                "Each lag column contains values from earlier "
                "observations, not future observations."
            ),
        ],
        color="light",
    )


# =========================================================
# Layout
# =========================================================

layout = dbc.Container(
    [
        # -------------------------------------------------
        # Page Title
        # -------------------------------------------------

        html.H2(
            "⏳ Lag Features",
            className="page-title",
        ),

        html.Hr(),

        # -------------------------------------------------
        # Introduction
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("What is a Lag Feature?"),

                    html.P(
                        """
                        A lag feature is a previous observation
                        from a time series that is used as a new
                        feature for analysis or prediction.
                        """
                    ),

                    html.P(
                        """
                        For example, if we want to predict this
                        month's sales, we can use last month's
                        sales as a feature.
                        """
                    ),

                    html.Div(
                        "Lag 1:  X(t - 1)  →  X(t)",
                        className="formula",
                    ),

                    html.P(
                        """
                        Here, X(t) represents the current value,
                        while X(t - 1) represents the value from
                        one time step earlier.
                        """
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Dataset and Controls
        # -------------------------------------------------

        dbc.Row(
            [
                # Dataset Information

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Dataset"),

                                html.P(
                                    "Monthly sales data for one year."
                                ),

                                html.P(
                                    "Each row represents one month."
                                ),

                                dbc.Badge(
                                    "12 Observations",
                                    color="primary",
                                    className="me-2",
                                ),

                                dbc.Badge(
                                    "Monthly Frequency",
                                    color="secondary",
                                ),
                            ]
                        ),
                        className="ts-card h-100",
                    ),
                    md=5,
                ),

                # Lag Selection

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Select Lag Features"),

                                html.P(
                                    """
                                    Choose one or more lag values.
                                    The selected lag columns will
                                    appear in the table.
                                    """
                                ),

                                dcc.Dropdown(
                                    id="lag-selector",

                                    options=[
                                        {
                                            "label": f"Lag {lag}",
                                            "value": lag,
                                        }
                                        for lag in range(1, 13)
                                    ],

                                    value=[1, 2, 3],

                                    multi=True,

                                    placeholder="Select lag values...",

                                    clearable=True,
                                ),

                                html.Small(
                                    """
                                    Tip: Start with Lag 1, Lag 2,
                                    and Lag 3 to see how the data shifts.
                                    """,
                                    className="text-muted",
                                ),
                            ]
                        ),
                        className="ts-card h-100",
                    ),
                    md=7,
                ),
            ],
            className="mb-4",
        ),

        # -------------------------------------------------
        # Lag Summary
        # -------------------------------------------------

        html.Div(
            id="lag-summary",
            className="mb-4",
        ),

        # -------------------------------------------------
        # Interactive Table
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "Original Data and Lag Features",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Each lag column contains values shifted
                        from earlier rows. Missing values appear
                        where there is no previous observation.
                        """
                    ),

                    dash_table.DataTable(
                        id="lag-table",

                        columns=[
                            {
                                "name": "Month",
                                "id": "Month",
                            },
                            {
                                "name": "Sales",
                                "id": "Sales",
                                "type": "numeric",
                            },
                        ],

                        data=sales_data[
                            ["Month", "Sales"]
                        ].to_dict("records"),

                        style_table={
                            "overflowX": "auto",
                        },

                        style_cell={
                            "textAlign": "center",
                            "padding": "10px",
                            "minWidth": "90px",
                            "width": "110px",
                            "maxWidth": "150px",
                        },

                        style_header={
                            "backgroundColor": "#16C4DD",
                            "fontWeight": "bold",
                        },

                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": "{Month} = 'Jan'",
                                },
                                "backgroundColor": "#F2F8FF",
                            }
                        ],

                        page_size=12,
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "How Do Lag Features Work?",
                        className="section-title",
                    ),

                    html.Div(
                        id="lag-explanation",
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
                        Pandas provides the shift() method to
                        create lag features. A positive shift
                        moves values down, making previous
                        observations available in the current row.
                        """
                    ),

                    html.Pre(
                        id="lag-code",
                        className="code-block",
                    ),
                ]
            ),
            className="ts-card mb-4",
        ),

        # -------------------------------------------------
        # How to Choose Lags
        # -------------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4(
                        "How Do We Choose the Right Lags?",
                        className="section-title",
                    ),

                    html.P(
                        """
                        Not every lag is useful for forecasting.
                        We can examine the relationship between
                        the current value and its past values
                        to identify potentially informative lags.
                        """
                    ),

                    html.H5("1. Autocorrelation Function (ACF)"),

                    html.P(
                        """
                        ACF measures the correlation between a
                        time series and its lagged values.
                        It includes both direct and indirect
                        relationships through intermediate lags.
                        """
                    ),

                    html.Div(
                        "ACF(k) = Corr(X(t), X(t - k))",
                        className="formula",
                    ),

                    html.P(
                        """
                        If the ACF at Lag 1 is high, it suggests
                        that the current values are correlated
                        with the values one time step earlier.
                        This can make Lag 1 a candidate feature.
                        """
                    ),

                    html.Hr(),

                    html.H5(
                        "2. Partial Autocorrelation Function (PACF)"
                    ),

                    html.P(
                        """
                        PACF measures the relationship between
                        the current value and a specific lag
                        after accounting for the effects of
                        the intermediate lags.
                        """
                    ),

                    html.Div(
                        "PACF(k) = Direct relationship at lag k",
                        className="formula",
                    ),

                    html.P(
                        """
                        For example, PACF at Lag 3 helps us
                        understand whether the third previous
                        observation contributes a relationship
                        with the current value beyond the effects
                        of Lags 1 and 2.
                        """
                    ),

                    html.Hr(),

                    html.H5("3. Practical Lag Selection"),

                    html.Ul(
                        [
                            html.Li(
                                "Start with Lag 1 to capture "
                                "the most recent observation."
                            ),

                            html.Li(
                                "Use ACF to identify lags with "
                                "potentially meaningful correlations."
                            ),

                            html.Li(
                                "Use PACF to examine the direct "
                                "relationship at individual lags."
                            ),

                            html.Li(
                                "For seasonal data, consider "
                                "seasonal lags such as Lag 12 "
                                "for monthly data."
                            ),

                            html.Li(
                                "Evaluate selected lags using "
                                "time-series validation and "
                                "forecasting performance."
                            ),
                        ]
                    ),

                    dbc.Alert(
                        """
                        Important: ACF and PACF help identify
                        candidate lags, but they do not guarantee
                        that a lag will improve forecasting.
                        The best lag selection depends on the
                        dataset, the model, and the forecasting task.
                        """,
                        color="info",
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
    Output("lag-table", "columns"),
    Output("lag-table", "data"),
    Output("lag-summary", "children"),
    Output("lag-explanation", "children"),
    Output("lag-code", "children"),

    Input("lag-selector", "value"),
)
def update_lag_dashboard(selected_lags):

    # Handle an empty selection.

    selected_lags = selected_lags or []

    # Generate the lagged dataset.

    lagged_data = create_lagged_data(
        sales_data,
        selected_lags,
    )

    # Create the table columns.

    columns = [
        {
            "name": "Month",
            "id": "Month",
        },
        {
            "name": "Sales",
            "id": "Sales",
            "type": "numeric",
        },
    ]

    for lag in sorted(selected_lags):

        columns.append(
            {
                "name": f"Lag {lag}",
                "id": f"Lag {lag}",
                "type": "numeric",
            }
        )

    # Convert NaN values to None for the DataTable.

    table_data = (
        lagged_data
        .astype(object)
        .where(pd.notna(lagged_data), None)
        .to_dict("records")
    )

    # Generate the summary.

    summary = create_lag_summary(selected_lags)

    # Generate the explanation.

    explanation = create_lag_explanation(selected_lags)

    # Generate the Python code.

    code = create_python_code(selected_lags)

    return (
        columns,
        table_data,
        summary,
        explanation,
        code,
    )