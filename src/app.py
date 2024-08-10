# Creates the interactive dashboard using the dash library
import pandas as pd
import dash

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
#from sklearn.ensemble import RandomForestClassifier
from PIL import Image

from prepare import PrepareData

# Initialize user_input_value
user_input_value = 1
prepared_data = PrepareData(download_new=False)
df = prepared_data.read_local_data('all',"data/prepared")
# Instantiate the dash app
app = dash.Dash(__name__)
server = app.server
# Create the app
# header
link = dbc.NavLink("View Github Repository", href="https://github.com/yourexodus/capstone_CDC")
link = dbc.NavLink("View Github Repository", href="https://github.com/yourexodus/capstone_CDC")
#### ********************************  ######
#############      BANNER ITEM   ############
#### ********************************  ######
banner_img_path = "assets/banner2.PNG"
banner_img = Image.open(banner_img_path).convert('RGB')

banner_item = dbc.Row(
    [
        dbc.Col(
            [
                dbc.CardImg(src=banner_img, style={'height': '200px', 'width': '100%'}),
                # Add other components for sidebar and navbar here...
            ]
        )
    ]
)


app.layout = html.Div([
    link,
    banner_item])
#----------------------------------------------#




if __name__ == '__main__':
    app.run_server(debug=True )


