import dash

# Creates the interactive dashboard using the dash library
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dash import dash_table
from dash_table import DataTable
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image

url = (
    "https://archive.ics.uci.edu/static/public/891/data.csv"
)


class PrepareData:
    """
    Downloads the data from cdc repository and applies several
    successive transformations to prepare it for modeling. The `run`
    method calls all the steps
    """

    def __init__(self, download_new=True):
        """
        Parameters
        ----------
        download_new : bool, determines whether new data will be downloaded
        or whether local saved data will be used
        """
        self.download_new = download_new

    def download_data(self):
        """
        Reads in a single dataset from the CDC website as csv

        Return: DataFrame
        """
        return pd.read_csv(url)

    def get_label_by_value(self, menu_income, value):
        """

          Args:
            menu_income: A list of dictionaries, each with 'label' and 'value' keys.
            value: The value to search for.

          Returns:
            The label corresponding to the given value, or None if not found.
          """

        for item in menu_income:
            if item['value'] == value:
                return item['label']
        return None

    def write_df_to_csv(self, df, name, directory, **kwargs):
        """
        Writes each raw data DataFrame to a file as a CSV

        Parameters
        ----------
        data : dictionary of DataFrames
        Returns
        -------X_test.to_csv('data/prepared/X_test_ScoreNewData.csv',index=False)
        None
        Note bashlash is a special character so will need to double the path string
        directory name, name of key.csv.  forwardign any keyword argument
        """
        df.to_csv(f'{directory}/{name}.csv', **kwargs)

    def write_data(self, data, directory, **kwargs):
        """
        Writes each raw data DataFrame to a file as a CSV

        Parameters
        ----------
        data : dictionary of DataFrames

        directory : string name of directory to save files i.e. "data/raw"

        kwargs : extra keyword arguments for the `to_csv` DataFrame method

        Returns
        -------X_test.to_csv('data/prepared/X_test_ScoreNewData.csv',index=False)
        None
        Note bashlash is a special character so will need to double the path string
        directory name, name of key.csv.  forwardign any keyword argument
        """
        for name, df in data.items():
            # df.to_csv(f'data/raw/{name}.csv')
            df.to_csv(f'{directory}/{name}_data.csv', **kwargs)

    def read_local_data(self, name, directory):
        """
          Read in one CSV as a DataFrame from the given directory

          Parameters
          ----------
          name : menhealth or menhealth or physical or dietary,heart,sex,edu,al

          directory : string name of directory to save files i.e. "data/raw"

          Returns
          -------
          DataFrame
          """
        return pd.read_csv(f"{directory}/{name}_data.csv")

    def run(self):
        """
          Run all cleaning and transformation steps

          Returns
          -------
          Dictionary of DataFrames
          """

        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}

        for i in names:
            if self.download_new:
                data[f"{i}"] = self.download_data()
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')

        return data

    def select_columns(self, df):
        """
          Selects fewer columns
          Parameters
          ----------
          df : DataFrame

          Returns
          -------
          df : DataFrame
          """
        sample_df = df
        cols = df.columns

        # choose few columns

        labels = ['Diabetes_binary', 'Gender', 'Types', 'MentHlth', 'GeneralHealth', 'Type',
                  'income', 'education', 'Sex', 'PhysHlth', 'PhysActivity', 'Fruits', 'HighBP',
                  'HighChol', 'Veggies', 'HeartDiseaseorAttack']

        filt = cols.isin(labels)

        return sample_df.loc[:, filt]

    def run2(self):
        """
          Run all cleaning and transformation steps

          Returns
          -------
          Dictionary of DataFrames
          """
        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}
        for i in names:

            if self.download_new:
                data[f"{i}"] = self.download_data()
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')

            df = self.select_columns(data[f"{i}"])  # step 1:  select column in data cleaning
            data[f"{i}"] = df
        return data

    def update_labels(self, df):
        """
          Replace a few of the area names using the REPLACE_AREA dictionary.

          Parameters
          ----------
          df : DataFrame

          Returns
          -------
          df : DataFrame
        """

        df['Gender'] = np.where(df['Sex'] == 0, 'men', 'women')
        df['Type'] = np.where(df['Diabetes_binary'] == 0, 'nondiabetic', 'diabetic')
        definitions = pd.Series([0, "Excellent", "Very good", "Good", "Fair", "Poor", "UNKNOWN"], dtype="category")

        reversefactor = dict(zip(range(7), definitions))
        df['GeneralHealth'] = np.vectorize(reversefactor.get)(df[['GenHlth']])
        definitions = pd.Series([0, "<10K", "10-15K", "15-20K", "20-25K", "25K-35K", "35-50K", "50-75K", "75>"],
                                dtype="category")
        reversefactor = dict(zip(range(9), definitions))

        df['income'] = np.vectorize(reversefactor.get)(df[['Income']])

        definitions = pd.Series([0, "None", "Grade1-8", "Grade9-11", "12orGED", "college1-3", "College4+"],
                                dtype="category")
        reversefactor = dict(zip(range(7), definitions))
        df['education'] = np.vectorize(reversefactor.get)(df[['Education']])

        return df

    def cdc_bar_plot2(self, df, x, y, t):

        """
        Dataframe, x, y axis
        Returns
        -------
        graph
        """
        title = ""
        type = ['nondiabetic' if t == 0 else 'diabetic']
        title = f'{x} vs Percentage for {type}'

        # graph shows the education condition of individuals in this dataset
        ge_df = self.group_data(df, x, y)
        ge_df = ge_df.sort_values(by='Count', ascending=True)
        y = ge_df["Percentage"]
        x = ge_df.index
        fig = go.Figure()
        fig.add_bar(x=x, y=y)
        fig.update_layout(height=400, width=800, title=title)

        return fig

    def run3(self):
        """
          Run all cleaning and transformation steps

          Returns
          -------
          Dictionary of DataFrames
          """
        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}
        for i in names:
            if self.download_new:
                data[f"{i}"] = self.download_data()
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')

            df = self.update_labels(data[f"{i}"])
            df = self.select_columns(df)  # step 1:  select column in data cleaning
            data[f"{i}"] = df
        return data

    def group_data(self, df, x, y):
        """
       Run all cleaning and transformation steps

       Returns
       -------
       counts and percentages of x,y in a DataFrames
       """

        # Assuming your DataFrame 'people' has columns 'GeneralHealth' and 'Type'

        # Filter data for relevant columns
        df = df[[x, y]]

        # Calculate counts using value_counts()
        counts = df[x].value_counts().to_frame(name="Count")

        # Calculate percentages (optional)
        # I want percentages:
        percentages = (counts["Count"] / len(df)) * 100
        counts["Percentage"] = percentages.apply("{:.1f}%".format)  # Format as percentages

        # Display the table
        return counts

    def run4(self):
        """
       Run all cleaning and transformation steps

       Returns
       -------
       Dictionary of DataFrames
       """
        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}
        for i in names:
            if self.download_new:
                data[f"{i}"] = self.download_data(1)
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')

            df = self.update_labels(df)  # step 1: update labels
            df = self.select_columns(df)  # step 2:  select column in data cleaning
            df = self.group_data(df, "GeneralHealth", "Type")  # step 4:
            data[f"{i}"] = df
        return data

    def run5(self):
        """
       Run all cleaning and transformation steps

       Returns
       -------
       Dictionary of DataFrames
       """
        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}
        for i in names:
            if self.download_new:
                data[f"{i}"] = self.download_data(1)
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')
            df = self.update_labels(df)  # step 1: update labels
            df = self.select_columns(df)  # step 2:  select column in data cleaning
            df = self.group_data(df, "income", "Type")  # step 3: group data
            data[f"{i}"] = df
        return data

    def run6(self):
        """
       Run all cleaning and transformation steps

       Returns
       -------
       Dictionary of DataFrames
       """
        names = ['menhealth', 'menhealth', 'physical', 'dietary', 'heart', 'sex', 'edu', 'all']
        data = {}
        for i in names:
            if self.download_new:
                data[f"{i}"] = self.download_data(1)
            else:
                data[f"{i}"] = self.read_local_data(i, 'data/raw')
            df = self.update_labels(df)  # step 1: update labels
            df = self.select_columns(df)  # step 2:  select column in data cleaning
            df = self.group_data(df, "education", "Type")  # step3: group data and count
            data[f"{i}"] = df
        return data

    def make_prediction(self, user_input):
        """
       Run all cleaning and transformation steps
       input X = df[['Income','GenHlth','MentHlth','PhysHlth','DiffWalk']]

       Returns
       -------
       Dictionary of DataFrames
       """
        cols = ['Income', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk']
        df = self.read_local_data('all', 'data/raw')
        y = df['Diabetes_binary']
        X = df[cols]
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        rfc = RandomForestClassifier(random_state=1234)
        rfc.fit(X_scaled, y)
        data = user_input
        reshaped_data = np.array(data).reshape(1, 5)
        prediction = rfc.predict(reshaped_data)

        if prediction[0] == 0:
            probability_pred = rfc.predict_proba(reshaped_data)[:, 0]
            result = probability_pred[0] * 100
            prt = f"You have a {result:.0f}% probability you will not be diagnosed with diabetes."
        else:
            probability_pred = rfc.predict_proba(reshaped_data)[:, 0]
            result = probability_pred[0] * 100
            prt = f"You have a {result:.0f}% probability you will be diagnosed with diabetes."

        return prt

    def graph_df(self, df, x, y):
        """
       input:dataframe, x, y
       Returns
       -------
       graph
       """

        df = df[[x, y]]
        t = f'{x} vs {y}'
        # Calculate counts and percentages
        counts = df[[x, y]].value_counts()
        percentages = (counts / len(df)) * 100
        # Create the bar chart
        plt.figure(figsize=(20, 6))
        ax = counts.plot(kind='bar')

        # Add percentage labels
        for i, v in enumerate(counts):
            ax.text(i, v + 0.1, f'{v} ({percentages[i]:.1f}%)', ha='center', va='bottom')

        plt.ylabel('Count')
        plt.xticks(fontsize=12)
        plt.title(t)
        plt.show()
        return plt

    def create_dataframe_counts_specificGenH(self, df, x, y, z):
        """Creates a DataFrame from the counts and percentages of two columns.

        Args:
            df: The original DataFrame.
            x: General health
            y: type
            z: Diabetes_binary
        Returns:
            A Pandas DataFrame containing counts, percentages, and the combined column.
        """
        filter1 = df['GeneralHealth'] == x
        filter2 = df['Type'] == y
        filter3 = df['Diabetes_binary'] == z

        test = df[filter1 & filter2 & filter3]  #
        counts = test[['Gender', 'income', 'education']].value_counts().reset_index(name='count')
        counts['percentage'] = (counts['count'] / len(test)) * 100
        counts['combine'] = x + ' and ' + y
        return counts

    def create_dataframe_counts_specificGenH_fig(self, df, x, y):
        """Creates a DataFrame from the counts and percentages of two columns.

        Args:
            df: The original DataFrame.
            x: General health
            y: type

        Returns:
            A Pandas DataFrame containing counts, percentages, and the combined column.
        """
        filter1 = df['GeneralHealth'] == x
        filter2 = df['Type'] == y
        if y == "nondiabetic":
            filter3 = df['Diabetes_binary'] == 0
        else:
            filter3 = df['Diabetes_binary'] == 1

        test = df[filter1 & filter2 & filter3]  #
        counts_gender = test[['Gender']].value_counts().reset_index(name='count')
        counts_gender['percentage'] = (counts_gender['count'] / len(test)) * 100

        counts_income = test[['income']].value_counts().reset_index(name='count')
        counts_income['percentage'] = (counts_income['count'] / len(test)) * 100

        counts_education = test[['education']].value_counts().reset_index(name='count')
        counts_education['percentage'] = (counts_education['count'] / len(test)) * 100
        ##########################  figure ####################################################################
        # Define color sets of paintings
        night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',
                        'rgb(36, 55, 57)', 'rgb(6, 4, 4)']
        sunflowers_colors = ['rgb(177, 127, 38)', 'rgb(205, 152, 36)', 'rgb(99, 79, 37)',
                             'rgb(129, 180, 179)', 'rgb(124, 103, 37)']
        irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                         'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
        cafe_colors = ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',
                       'rgb(175, 51, 21)', 'rgb(35, 36, 21)']

        # Convert color lists to dictionaries
        night_colors_dict = dict(zip(range(len(night_colors)), night_colors))
        sunflowers_colors_dict = dict(zip(range(len(sunflowers_colors)), sunflowers_colors))
        irises_colors_dict = dict(zip(range(len(irises_colors)), irises_colors))

        # Define pie charts

        # Create subplots, using 'domain' type for pie charts
        specs = [[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]]
        fig = make_subplots(rows=2, cols=2, specs=specs)

        # Define pie charts
        fig.add_trace(go.Pie(labels=counts_gender['Gender'], values=counts_gender['percentage'], name='Gender',
                             marker_colors=night_colors), 1, 1)
        fig.add_trace(go.Pie(labels=counts_income['income'], values=counts_income['percentage'], name='Income',
                             marker_colors=sunflowers_colors), 1, 2)
        fig.add_trace(
            go.Pie(labels=counts_education['education'], values=counts_education['percentage'], name='Education',
                   marker_colors=irises_colors), 2, 1)

        # Tune layout and hover info
        fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
        fig.update_traces(hoverinfo='label+percent+name', textinfo='none')
        fig.update_layout(
            height=600,  # Increase height
            width=800

        )
        fig.update(layout_title_text=f'Breakdown Distribution of {y.capitalize()} in {x.capitalize()} health',
                   layout_showlegend=True)

        fig = go.Figure(fig)
        return fig

    def create_dataframe_from_counts_part2(self, df, x, y):
        """ Creates a DataFrame from the counts and percentages of two columns.

        Args:
            df: The original DataFrame.
            x: The first column to group by.
            y: The second column to group by.

        Returns:
            A Pandas DataFrame containing counts, percentages, and the combined column.
        """

        counts = df[[x, y]].value_counts().reset_index(name='count')
        counts.columns = [x, y, 'count']
        percentage = (counts['count'] / len(df)) * 100
        counts["percentage"] = percentage.apply("{:.1f}%".format)  # Format as percentages
        counts['combined'] = counts[x] + ' and ' + counts[y]
        return counts

    def group_data(self, df, x, y):
        """
        Run all cleaning and transformation steps

        Returns
        -------
        counts and percentages of x,y in a DataFrames
        """

        # Assuming your DataFrame 'people' has columns 'GeneralHealth' and 'Type'

        # Filter data for relevant columns
        df = df[[x, y]]

        # Calculate counts using value_counts()
        counts = df[x].value_counts().to_frame(name="Count")

        # Calculate percentages (optional)
        # I want percentages:
        percentages = (counts["Count"] / len(df)) * 100
        counts["Percentage"] = percentages.apply("{:.1f}%".format)  # Format as percentages

        # Display the table
        return counts

    def cdc_bar_plot_combined_pie(self, df, x, y, t):

        """  test
        Dataframe, x, y axis
        Returns
        -------
        graph
        """
        summary = self.create_dataframe_from_counts_part2(df, x, y)

        fig = go.Figure()

        # Prepare data with combined values for both labels and values
        data = [go.Pie(labels=summary['combined'], values=summary['percentage'])]

        # Add the trace to the figure
        fig.add_trace(*data)

        fig.update_layout(title='Distribution of Health by Type')

        return fig

app = dash.Dash(__name__)
#app = dash.Dash(
#    __name__,
#    # stylesheet for dash_bootstrap_components
#    external_stylesheets=[
#        "https://stackpath.bootstrapcdn.com/bootswatch/4.5.0/flatly/bootstrap.min.css"
#    ],
#)
server = app.server
# Initialize user_input_value
user_input_value = 1

##########################################################################################
#################      READ LOCAL DATA :  ALL   ##########################################
##########################################################################################
prepared_data = PrepareData(download_new=False)
df = prepared_data.read_local_data('all', "data/prepared")
##########################################################################################


#########################################################################################
########## Header Section Divs: link, Banner, mytable:                              #####
##########                                                                          #####
##########               mytable items:     doctorcat_item and meowmidwest_item     #####
#########################################################################################
#### ********************************  ######
#############      LINK       ################
#### ********************************  ######

link = dbc.NavLink("View Github Repository", href="https://github.com/yourexodus/capstone_CDC")
#### ********************************  ######
#############      BANNER ITEM   ############
#### ********************************  ######
banner_img_path = "src/assets/banner2.PNG"
banner_img = Image.open(banner_img_path)

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

######### ******************************************  ##############
#########              mytable ITEMs:                  ##############
#########      doctorcat_item and meowmidwest_item    ##############
######### ********************************  ########################
doctorcat_img_path = "src/assets/doctorcat.png"
doctorcat_img = Image.open(doctorcat_img_path)

doctorcat_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([

                        html.Img(src=doctorcat_img,
                                 style={'width': '100%', 'height': '500px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)
doctorcat_item.style = {'gridArea': "doctorcat_item"}

# Update the video element to use the get_video_frame function
meowmidwest_img_path = "src/assets/MeowMidwest.gif"
meowmidwest_item = html.Div(
    [
        html.Img(src=meowmidwest_img_path, alt="Meow Midwest", style={"width": "550px", "height": "500px"})
    ]
)

meowmidwest_item.style = {'gridArea': "meowmidwest_item"}
# Define table header and data
header = html.Thead(
    html.Tr([html.Th("Midwest Meow Hospital hours: Sun-up to Sun-down")])  # Single header row with a single column
)

data_row = html.Tr([html.Td(doctorcat_item), html.Td(meowmidwest_item)])

# Create the table
mytable = html.Table([data_row])

#########################################################################################
#########################################################################################
##########                         Dropdown Section                                 #####
##########                          defined in the layout                           #####
##########                          added dropdown list values here                 #####
#########################################################################################
#########################################################################################

menu_income = [
    {'label': '1 - Less than $10,000', 'value': 1},
    {'label': '2 - Less than $15,000 ($10,000 to less than $15,000)', 'value': 2},
    {'label': '3 - Less than $20,000 ($15,000 to less than $20,000)', 'value': 3},
    {'label': '4 - Less than $25,000 ($20,000 to less than $25,000)', 'value': 4},
    {'label': '5 - Less than $35,000 ($25,000 to less than $35,000)', 'value': 5},
    {'label': '6 - Less than $50,000 ($35,000 to less than $50,000)', 'value': 6},
    {'label': '7 - Less than $75,000 ($50,000 to less than $75,000)', 'value': 7},
    {'label': '8 - $75,000 or more', 'value': 8}
]

gen_health = [
    {'label': '1 Excellent', 'value': 1},
    {'label': '2 Very good', 'value': 2},
    {'label': '3 Good', 'value': 3},
    {'label': '4 Fair', 'value': 4},
    {'label': '5 Poor', 'value': 5},
    {'label': '7 Don’t know/Not Sure', 'value': 7},
    {'label': '9 Refused', 'value': 7}

]
# number of days mental health not good
men_health = [
    {'label': '1: 1 - 30 Number of days ', 'value': 1},
    {'label': '2: 88 None ', 'value': 2},
    {'label': '3: 77 Don’t know/Not sure ', 'value': 3},
    {'label': '4: 99 Refused', 'value': 4},
    {'label': '5: BLANK Not asked or Missing', 'value': 5},

]
# numger of days phyical health not good
phy_health = [
    {'label': '1: 1 - 30 Number of days ', 'value': 1},
    {'label': '2: 88 None ', 'value': 2},
    {'label': '3: 77 Don’t know/Not sure ', 'value': 3},
    {'label': '4: 99 Refused', 'value': 4},
    {'label': '5: BLANK Not asked or Missing', 'value': 5},

]

# numger of days you have difficulty walking or climbing stairs
diff = [
    {'label': '1: 1 - 30 Number of days ', 'value': 1},
    {'label': '2: 88 None ', 'value': 2},
    {'label': '3: 77 Don’t know/Not sure ', 'value': 3},
    {'label': '4: 99 Refused', 'value': 4},
    {'label': '5: BLANK Not asked or Missing', 'value': 5},

]


#############################################################################
################## Layout Diff:  will hold the result of the prediction #####

############################################################################
################## Layout: Prediction VALUE message ########################
############################################################################


############################################################################
##################  RAW TABLE      ########################################
###########################################################################
# get the data
def create_raw_table(raw):
    df = raw.head()

    columns = []
    # columns = [{"name": "generalhealth_type",
    #            "id": "gentype", "type": "text"}]
    for name in raw.columns:
        col_info = {
            "name": name,
            "id": name,
            "type": "text",
            "format": {'specifier': ','}
        }
        columns.append(col_info)

    data = df.to_dict("records")
    return DataTable(
        id="raw-table",  # raw-table
        columns=columns,
        data=data,
        active_cell={"row": 0, "column": 0},
        fixed_rows={"headers": True},
        sort_action="native",
        derived_virtual_data=data,
        style_table={
            "minHeight": "30vh",
            "height": "40vh",
            "overflowX": "scroll",
            "borderRadius": "0px 0px 10px 10px",
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "fontFamily": "verdana",
            "width": "50px",

        },
        style_header={
            "textAlign": "center",
            "fontSize": 14,
        },
        style_data={
            "fontSize": 12,
        },
        style_data_conditional=[
            {
                "if": {"column_id": "gentype"},
                "width": "420px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",
            },

            {
                "if": {"column_id": "index"},
                "width": "50px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",

            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )


raw = prepared_data.read_local_data('all', 'data/raw')

#################################################
### call create_table_method defined in prepare.py
###############################################
raw_table = create_raw_table(raw)

############################################################################
################   BORDER ITEM  ###########################################
###########################################################################

border1_img_path = "src/assets/border1.png"
border1_img = Image.open(border1_img_path)

border1_item = dbc.Row(
    [
        dbc.Col(
            [
                dbc.CardImg(src=border1_img, style={'width': '100%'}),
                # Add other components for sidebar and navbar here...
            ]
        )
    ]
)
border1_item.style = {'gridArea': "border1_item"}
############################################################################
################   Bullet Points  : html.Li(step) for step in steps  #######
###########################################################################


# Bullet Point data

steps = [
    "Import pandas",
    "Reading in Data",
    "Selected Columns",
    "Reset Index",
    "Identified no missing data",
    "Relationships between variables uisng heatmap"
]

tools_used = [
    "Git - source code management",
    "GitBash - terminal application used to push changes up to Github Repository",
    "Jupyter Notebook - web application used to create my documents",
    "Anaconda Prompt - used command line interface to manage my virutal environment and access Jupyter notebook",
    "TCPView - Used to identify and terminate apps running ports on local machine",
    "Pycharm -  Integrated Development Environment (IDE) used to launch my app to render",
    "Render - free web hosting service used to deploy my app to the web"

]

issues = [

    "box plot shows large variances and outliers in Mental Health and Physical Health Data.   outliers can be removed from the dataset prior to modeling. It is good practice to note specifically what outlier values were removed and why",
    "Outliers can be removed from the dataset prior to modeling. It is good practice to note specifically what outlier values were removed and why",
    " Data was note  data should form a bell shaped curve but skewed. How will you transform the skewed data so it is suitable for modeling"

]

treatment = [

    "1-Address Outliers using IQR method",
    "2-Replace codes with label for better interprepation of data",
    "3-Aggregrate data for graph"
]

#########################################################################################
##################    Exploring the Data SECTION ########################################
################                                          ##############################
################     exploredata_item, heatmap_item, boxplot_item       #################
#########################################################################################

#############################################################
################     exploredata_item        #################
###############################################################
exploredata_img_path = "src/assets/exploredata.png"
exploredata_img = Image.open(exploredata_img_path)
exploredata_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        # html.Img(src=banner_img, style={'width': '100%', 'height': '50%'})
                        html.Img(src=exploredata_img,
                                 style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

exploredata_item.style = {'gridArea': "exploredata_item"}

############################################################
################     heatmap_item       ######################
###############################################################

heatmap_img_path = "src/assets/heatmap.PNG"
heatmap_img = Image.open(heatmap_img_path)
heatmap_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        # html.Img(src=banner_img, style={'width': '100%', 'height': '50%'})
                        html.Img(src=heatmap_img,
                                 style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

heatmap_item.style = {'gridArea': "heatmap_item"}

############################################################
################     boxplot_item       ######################
###############################################################
boxplot_img_path = "src/assets/boxplot.PNG"
boxplot_img = Image.open(boxplot_img_path)
boxplot_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        html.Img(src=boxplot_img,
                                 style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                        'align-items': 'center'})
                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

boxplot_item.style = {'gridArea': "boxplot_item"}

#########################################################################################
##################    Feature Engineered & Aggregated data SECTION        ###############
##################                 outliers_item , updatecolumns , updated_table ########
#########################################################################################

############################################################
################     outliers_item       ######################
###############################################################
outliers_img_path = "src/assets/outliers.png"
outliers_img = Image.open(outliers_img_path)
outliers_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        # html.Img(src=banner_img, style={'width': '100%', 'height': '50%'})
                        html.Img(src=outliers_img,
                                 style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

outliers_item.style = {'gridArea': "outliers_item"}

############################################################
################     updatecolumns_item       ################
###############################################################
updateColumns_img_path = "src/assets/updateColumns.png"
updateColumns_img = Image.open(updateColumns_img_path)

updatecolumns_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        # html.Img(src=banner_img, style={'width': '100%', 'height': '50%'})
                        html.Img(src=updateColumns_img,
                                 style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

updatecolumns_item.style = {'gridArea': "updatecolumns_item"}


##############################################################
################     updated table       ######################
###############################################################
def create_updated_table(df):
    df = df.head()

    columns = []

    for name in df.columns:
        col_info = {
            "name": name,
            "id": name,
            "type": "text",
            "format": {'specifier': ','}
        }
        columns.append(col_info)

    data = df.to_dict("records")
    return DataTable(
        id="updated-table",  # updated-table
        columns=columns,
        data=data,
        active_cell={"row": 0, "column": 0},
        fixed_rows={"headers": True},
        sort_action="native",
        derived_virtual_data=data,
        style_table={
            "minHeight": "30vh",
            "height": "40vh",
            "overflowX": "scroll",
            "borderRadius": "0px 0px 10px 10px",
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "fontFamily": "verdana",
            "width": "50px",

        },
        style_header={
            "textAlign": "center",
            "fontSize": 14,
        },
        style_data={
            "fontSize": 12,
        },
        style_data_conditional=[
            {
                "if": {"column_id": "gentype"},
                "width": "420px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",
            },

            {
                "if": {"column_id": "index"},
                "width": "50px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",

            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )


df_prep = prepared_data.read_local_data('all', "data/prepared")
updated_table = create_updated_table(df_prep)

#########################################################################################
##################    Aggregate and view Bar chart percentage   #########################
####                                                                                ####
####          graph_01, summary_table,(id='graph-output,analysis_graph_figure       ####
####                                                                                ###
#########################################################################################

############################################################
################     graph_01               ################
##############################################################
df = prepared_data.read_local_data('all', 'data/prepared')
summary = prepared_data.create_dataframe_from_counts_part2(df, 'GeneralHealth', 'Type')
summary = summary.reset_index()

circle_fig = px.pie(summary, values='count', names='percentage', title='Distribution of Health by Type')  # No filtering

graph_01 = dcc.Graph(figure=circle_fig, style={'gridArea': "graph_01"})


##############################################################
################     summary_table           ################
#############################################################
def create_sum_table(summary):
    used_columns = ["index", "combined", "count", "percentage", "GeneralHealth", "Type"]
    df = summary[used_columns]
    df = df.rename(columns={"combined": "generalhealth_type", "count": "total"})
    columns = []
    # columns = [{"name": "generalhealth_type",
    #            "id": "gentype", "type": "text"}]
    for name in df.columns:
        col_info = {
            "name": name,
            "id": name,
            "type": "text",
            "format": {'specifier': ','}
        }
        columns.append(col_info)

    data = df.sort_values("total", ascending=False).to_dict("records")
    return DataTable(
        id="sum-table",  # sum-table
        columns=columns,
        data=data,
        active_cell={"row": 0, "column": 0},
        fixed_rows={"headers": True},
        sort_action="native",
        derived_virtual_data=data,
        style_table={
            "minHeight": "80vh",
            "height": "40vh",
            "overflowY": "scroll",
            "borderRadius": "0px 0px 10px 10px",
        },
        style_cell={
            "whiteSpace": "normal",
            "height": "auto",
            "fontFamily": "verdana",
        },
        style_header={
            "textAlign": "center",
            "fontSize": 14,
        },
        style_data={
            "fontSize": 12,
        },
        style_data_conditional=[
            {
                "if": {"column_id": "gentype"},
                "width": "420px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",
            },

            {
                "if": {"column_id": "index"},
                "width": "70px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer",
            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )


summary_table = create_sum_table(summary)

############################################################
################     graph-output      -- OUTPUT     ################
###############################################################


##############################################################
################     analysis_graph_figure      ##############
###############################################################

analysis_graph = prepared_data.create_dataframe_counts_specificGenH_fig(df, 'Very good', 'diabetic')

analysis_graph_figure = dcc.Graph(figure=analysis_graph, id="analysis_graph_figure",
                                  style={'gridArea': "analysis_graph_figure"})

#########################################################################################
##################    PREDICTION MODELING SECTION #######################################
#########################################################################################

Model_img_path = "src/assets/Model.png"
Model_img = Image.open(Model_img_path)

Model_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        html.Img(src=Model_img, style={'width': '800px', 'height': '600px', 'justify-content': 'center',
                                                       'align-items': 'center'})

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)

Model_item.style = {'gridArea': "Model_item"}

#########################################################################################
#######     Create Test program to accept user input and display prediction SECTION  ####
######                        programlink, mytable2  : prediction code.PNG and code.mp4 #
#########################################################################################

programlink = html.A('Python Program Making Prediction',
                     href="https://github.com/yourexodus/capstone_CDC/blob/4b4f4f3c0933f6968cb9b2651c8c35f3f5372d1f/Prediction_Menu.py")

##############################################
predictionCode_img_path = "src/assets/predictionCode.png"
predictionCode_img = Image.open(predictionCode_img_path)
predictionCode_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([

                        html.Img(src=predictionCode_img,
                                 style={'width': '100%', 'height': '500px', 'justify-content': 'center',
                                        'align-items': 'center'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)
predictionCode_item.style = {'gridArea': "predictionCode_item"}
##############################################################
code_item = html.Div(
    [
        html.Img(src="assets/PredictionProgram.gif", alt="predprogram", style={"width": "800px", "height": "450px"})
    ]

)


code_item.style = {'gridArea': "code_item"}

##############################################################
################     mytable2                   ##############
###############################################################

# Define table header and data
header = html.Thead(
    html.Tr([html.Th("Test")])  # Single header row with a single column
)

data_row2 = html.Tr([html.Td(predictionCode_item), html.Td(code_item)])  # Single data row with two cells

# Create the table
mytable2 = html.Table([data_row2])

########################################################################################
##############         TESTING PROGRAM                    ##############################
#########################################################################################

flowchart_img_path = "src/assets/flowchart.png"
flowchart_img = Image.open(flowchart_img_path)

flowchart_item = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.Div([
                        # html.Img(src=banner_img, style={'width': '100%', 'height': '50%'})
                        html.Img(src=flowchart_img, style={'width': '500px', 'height': '300px'})
                        # html.Img(src=banner_img, 'width': '50%', 'height': '200px'),               # using the pillow image variable

                    ]),
                    html.Div(className="sidebar-wrapper"),
                ]
            ),
            className="sidebar",
        ),
        html.Div(
            html.Div(
                html.Div(className="container-fluid"),
                className="navbar navbar-expand-lg navbar-transparent navbar-absolute fixed-top ",
            ),
            className="main-panel",
        ),
    ]
)
flowchart_item.style = {'gridArea': "flowchart_item"}
#########################################################################################

#########################################################################################
##################    LAYOUT SECTION ####################################################
#########################################################################################

# layout is saved to a variable so I dont have to keep running it
app.layout = html.Div([
    link,
    banner_item,
    mytable,  # add doctor cat
    ########################################################
    ############# Prediction output ######################
    html.Div(html.H2("Questionaire"))
    ,
    html.Br()
    ,
    html.Div(

        children=[
            html.A(
                "Note: drop downs are in a persistence state.  click new values in all the fields to populate a prediction.  The last field will calls the prediction.  It can take up to 4 minutes  to display")
        ])
    ,
    ########################################################################################################
    ################## Define input and out for income drop down ###########################################
    #######################################################################################################
    html.Br()
    ,
    html.Div(
        children=[
            'Please choose your income range.:', dcc.Dropdown(
                id='menu_income_id',
                options=menu_income,
                placeholder="Please choose your income range.",
                value=user_input_value,
                persistence=True,  # store user dropdown
            )
        ],
        style={
            "display": "block"
        }

    ),


    ######################################################################
    ##################  OUTPUT VALUE for income #######################

    html.Div(id='income-output'),
    html.Br(),
    #################################################################################################
    ###########  gen_health  ########################################################################
    ################################################################################################
    html.Div(  # Assuming user_input1 is a Div
        children=[
            "Please enter your general health code:", dcc.Dropdown(
                id='gen_health_id',
                options=gen_health,
                placeholder="Please enter your general health code.",
                value=user_input_value,
                persistence=True,  # store user dropdown
            )
        ],
        style={
            "display": "block",  # Set display to block
        }
    ),
    ######################################################################
    ##################  OUTPUT VALUE for general health #######################

    html.Div(id='gen-health-output'),
    html.Br(),
    ################################################################################################
    html.Div(
        children=[
            "Please choose your physical health code:", dcc.Dropdown(
                id='phy_health_id',
                options=phy_health,
                placeholder="Please choose your physical health code.",
                value=user_input_value,
                persistence=True,  # store user dropdown
            )
        ],
        style={
            "display": "block"
        }
    ),
    ######################################################################
    ##################  OUTPUT VALUE for physical health days #######################

    html.Div(id='phy-health-output'),
    html.Br(),
    ################################################################################################
    html.Div(
        children=[
            "Please choose your physical health range:", dcc.Dropdown(
                id='men_health_id',
                options=men_health,
                placeholder="Please choose your physical health range.",
                value=user_input_value,
                persistence=True,  # store user dropdown
            )
        ],
        style={
            "display": "block"
        }
    ),
    ######################################################################
    ##################  OUTPUT VALUE for mental health #######################

    html.Div(id='diff-output'),
    html.Br(),
    #################################
    html.Div(
        children=[
            "Please choose number of days you had diffculty walking:", dcc.Dropdown(
                id='diff_id',
                options=diff,
                placeholder="Please choose number of days you had diffculty walking.",
                value=user_input_value,
                persistence=True,  # store user dropdown
            )
        ],
        style={
            "display": "block"
        }
    ),
    html.Br(),
    ######################################################################
    ##################  OUTPUT VALUE for income #######################
    html.Div(
        children=[
            html.H2(id='diff-output')
        ])

    ,
    html.Br()
    ,
    html.Div(id='prediction-output')
    ,

    html.Br(),
    html.Br(),
    ################################
    border1_item,
    ###############################################
    #########  About the Data ######################
    ##############################################

    html.Div(
        children=[
            html.H1("Capstone: CDC Diabetes Project")
        ])
    ,
    html.Div(
        children=[
            html.P(
                "This is the result of a Logcial Regression Capstone Project used to determine the probability of diabetes based select features.  My desire is to use this capstone to demonstrate skills in the areas of data science, machine learning, and python to predict and outcome using data analysis.")
        ])
    ,
    html.Div(
        children=[
            html.H2("About the Data"),
            html.P(
                '"The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. -- source: Machine learning repository"'),
            html.A('Available Data', "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"),
            html.A('CDC data Codes', "https://www.cdc.gov/brfss/annual_data/2014/pdf/CODEBOOK14_LLCP.pdf"),

        ])
    ,
    html.Div(
        children=[
            html.H3("Raw data")
        ])
    ,
    html.Div([raw_table])
    ,
    border1_item
    ,
    html.Div(
        children=[
            html.H2("Step 1: Explored the Data")
        ])
    ,

    html.Div([
        html.Ul([
            html.Li(step) for step in steps
        ])
    ])

    ,
    html.Div(
        children=[
            html.H3("Identified issues with the data")
        ])

    ,

    html.Div([exploredata_item, heatmap_item])
    ,
    html.Div([
        html.Ul([
            html.Li(step) for step in issues
        ])
    ])
    ,

    html.Div([boxplot_item])

    ,
    html.Div(
        children=[
            html.A(
                "box plot shows large variances and outliers in Mental Health and Physical Health Data. This can affect my model performance.  Also, since the data is categorical data, I will use a categorical response to determine the probability of diabetes based the most prominent features identified in my heatmap. According to the stats, there is a 30point difference from min and max for Mental and Physical Health data")
        ])

    ,
    border1_item
    ,
    html.Div(
        children=[
            html.H2("Step 2: Feature Engineered & Aggregated data")
        ])
    ,

    html.Div(
        children=[
            html.H3("Purpose of this Section:")
        ])
    ,
    html.Div([
        html.Ul([
            html.Li(step) for step in treatment
        ])
    ])

    ,

    html.Div(
        children=[
            html.H3("1. Address Outliers using IQR method")
        ])

    ,

    html.Div([outliers_item])

    ,

    html.Div(
        children=[
            html.A(
                "I identied 14% percent of the records are outliers using IQP the lower & upper bounds in the MentHlth data.  I identied 19% percent of the records are outliers using IQP the lower & upper bounds in the PhysHlth data")
        ])

    ,
    html.Br()
    ,
    html.Div(
        children=[
            html.H3("3-Replace codes with label for better interprepation of data")
        ])

    ,
    html.Div([updatecolumns_item])

    ,
    html.Div(
        children=[
            html.A("Top 4 header rows of updated data table")
        ])
    ,
    html.Div([updated_table])
    ,
    html.Br()
    ,
    html.Div(
        children=[
            html.H3("2- Aggregate and view Bar chart percentage")
        ])
    ,

    html.Div([graph_01])
    ,
    html.Br()
    ,
    html.Div(
        children=[
            html.A(
                "Each row selected in the table below will provide insight on the percentages in the circle graph above.  For furher insight you can select a row the 3 interactive Graphs below will update .  If you hoover over the graphs you can see more information about the  selected population")
        ])

    ,
    html.Br()
    ,
    html.Div([summary_table])
    ,
    html.Div(id='graph-output')
    ,
    html.Div([analysis_graph_figure])
    ,
    border1_item

    ,

    html.Div(
        children=[
            html.H2("Step 3: Create Predictive models")
        ])
    ,
    html.Br()
    ,
    html.Div(
        children=[
            html.H3("Nominal Classification Problem: Predict diabetis for a customer.")
        ])
    ,
    html.Div(
        children=[
            html.A(
                "Tested out several feature combination using Test Train Split method. Select the best accuracy score"),

        ])

    ,
    html.Div(
        children=[
            html.A("Feature columns are cols =['Income','GenHlth','MentHlth','PhysHlth','DiffWalk']"),

        ])

    ,
    html.Div(
        children=[
            html.H4("Target column is y  = df['Diabetes_binary']")
        ])
    ,

    html.Div([Model_item])
    ,
    border1_item
    ,

    html.Div(
        children=[
            html.H2("Step 4: Create Test program to accept user input and display prediction")
        ])
    ,
    programlink
    ,
    mytable2  # html.Div(programvideo_div)
    ,
    border1_item
    ,

    html.Div(
        children=[
            html.H2("Step 5: Build the app")
        ])
    ,
    html.Div([flowchart_item])
    ,

    border1_item
    ,
    html.Div(
        children=[
            html.H2("Summary Tools Used")
        ])
    ,

    html.Div([
        html.Ul([
            html.Li(step) for step in tools_used
        ])
    ])

    ,
], style={'padding': 10})


#########################################
############ call back for income drop down
###############################

@app.callback(
    Output('income-output', 'children'),
    [Input('menu_income_id', 'value')]

)
def callback_a(income_value):
    # Access and use the global variable here
    # For example:

    return f"You've selected: {income_value}"


#################################################3
########### call back for gen health ###########
#########################
@app.callback(
    Output('gen-health-output', 'children'),
    [Input('gen_health_id', 'value')]
)
def callback_b(gen_health_value):
    gl_gen_health = gen_health_value
    return 'Youve selected "{}"'.format(gen_health_value)


#################################################
###### call back for phy health
############################################
@app.callback(
    Output('phy-health-output', 'children'),
    [Input('phy_health_id', 'value')]
)
def callback_c(phy_health_value):
    gl_phy_health = phy_health_value
    return 'Youve selected "{}"'.format(phy_health_value)


#################################################
###### call back for phy health
############################################
@app.callback(
    Output('men-health-output', 'children'),
    [Input('men_health_id', 'value')]
)
def callback_d(men_health_value):
    gl_men_health = men_health_value
    return 'Youve selected "{}"'.format(men_health_value)
@app.callback(
    Output('diff-output', 'children'),
    [Input('diff_id', 'value')]
)
def callback_d(diff_value):
    gl_diff = diff_value
    return 'Youve selected "{}"'.format(diff_value)

#################################################
###### call back for phy health
############################################

@app.callback(
    Output('prediction-output', 'children'),
    [Input('combined_input', 'value')],  # Combine all inputs into a single input
    prevent_initial_call=True
)
def callback_e(combined_input):
    if not all(combined_input.values()):
        return 'Youve selected "{}"'.format(combined_input['diff_value'])

    # Extract values from the combined input
    menu_income_id = combined_input['menu_income_id']
    gen_health_id = combined_input['gen_health_id']
    phy_health_id = combined_input['phy_health_id']
    men_health_id = combined_input['men_health_id']
    diff_value = combined_input['diff_value']

    all_input_data = [menu_income_id, gen_health_id, phy_health_id, men_health_id, diff_value]
    result = prepared_data.make_prediction(all_input_data)
    return result

    #return result + ".  Refresh your browser to start again"


###########################################################################
####### Call back for graphs
########################################################################
@app.callback(
    Output("analysis_graph_figure", "figure"),
    Input('sum-table', 'active_cell'),
    State('sum-table', 'derived_virtual_data')
)
def change_area_graphs(sum_cell, sum_data):
    ##    """
    ##  Change the all three graphs in the upper right hand corner of the app
    #   Parameters
    #    ----------
    ##    avg_cell : dict with keys `row` and `cell` mapped to integers of cell location
    #   avg_data : list of dicts of one country per row.
    #                    Has keys Country, Deaths, Cases, Deaths per Million, Cases per Million
    #   Returns
    #    -------
    #    List of three plotly figures, one for each of the `Output`
    #    """
    row_number = sum_cell["row"]
    row_data = sum_data[row_number]

    fig = prepared_data.create_dataframe_counts_specificGenH_fig(df, row_data["GeneralHealth"], row_data["Type"])
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

