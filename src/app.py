# Creates the interactive dashboard using the dash library
import pandas as pd
import dash

from dash_table import DataTable
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from dash import html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Instantiate the dash app
app = dash.Dash(
    __name__
)
