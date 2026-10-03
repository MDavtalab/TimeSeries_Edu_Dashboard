"""
=========================================================
Time Series Learning Lab (TSLL)
Configuration
=========================================================
"""

import dash_bootstrap_components as dbc

# -------------------------------------------------------
# Application
# -------------------------------------------------------

APP_NAME = "Time Series Learning Lab"
APP_VERSION = "0.1.0"
AUTHOR = "Mohammad Davtalab & ChatGPT"

# -------------------------------------------------------
# Theme
# -------------------------------------------------------

THEME = dbc.themes.BOOTSTRAP

# -------------------------------------------------------
# Sidebar Pages
# -------------------------------------------------------

PAGES = [

    {
        "icon": "🏠",
        "title": "Home",
        "path": "/",
        "enabled": True,
    },

    {
        "icon": "📈",
        "title": "Moving Average",
        "path": "/moving-average",
        "enabled": True,
    },

    {
        "icon": "⏳",
        "title": "Lag Features",
        "path": "/lag-features",
        "enabled": True,
    },

    {
        "icon": "📉",
        "title": "Trend",
        "path": "/trend",
        "enabled": True,
    },

    {
        "icon": "🔄",
        "title": "Seasonality",
        "path": "/seasonality",
        "enabled": True,
    },

    {
        "icon": "📊",
        "title": "Time Series as Features",
        "path": "/time-series-features",
        "enabled": True,
    },
    {
        "icon": "🖇",
        "title": "Hybrid Models",
        "path": "/hybrid-models",
        "enabled": True,
    },
    {
        "icon": "⚙️",
        "title": "Forecasting With Machine Learning",
        "path": "/forecasting-with-machine-learning",
        "enabled": True,
    },
]