import numpy as np
import pandas as pd
from dash.dash_table import DataTable
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 

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
        df.to_csv(f'{directory}/{name}.csv',**kwargs)
        
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
                  'income', 'education', 'Sex', 'PhysHlth', 'PhysActivity', 'Fruits','HighBP',   
                  'HighChol','Veggies', 'HeartDiseaseorAttack']

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

            df = self.select_columns(data[f"{i}"] )  # step 1:  select column in data cleaning
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

        definitions = pd.Series([0, "None", "Grade1-8", "Grade9-11", "12orGED", "college1-3", "College4+"], dtype="category")
        reversefactor = dict(zip(range(7), definitions))
        df['education'] = np.vectorize(reversefactor.get)(df[['Education']])

        return df
    def cdc_bar_plot2(self,df,x,y,t):
    
        """
        Dataframe, x, y axis
        Returns
        -------
        graph
        """ 
        title=""
        type = ['nondiabetic' if t == 0 else 'diabetic']
        title = f'{x} vs Percentage for {type}'
        
        #graph shows the education condition of individuals in this dataset
        ge_df = self.group_data( df,x,y)
        ge_df = ge_df.sort_values(by='Count', ascending=True)
        y = ge_df["Percentage"]
        x= ge_df.index
        fig = go.Figure()
        fig.add_bar(x=x, y=y)
        fig.update_layout(height=400,width=800,title=title)                  
                       
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
            df = self.group_data(  df, "GeneralHealth", "Type")  # step 4:
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
            df = self.group_data( df, "income", "Type")  # step 3: group data
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
        filter1  = df['GeneralHealth'] == x
        filter2 = df['Type'] == y
        filter3 = df['Diabetes_binary'] == z
        
        test = df [filter1 & filter2 & filter3] #
        counts  = test[['Gender','income','education']].value_counts().reset_index(name='count')
        counts['percentage'] = (counts['count'] / len(test)) * 100 
        counts['combine'] =  x + ' and ' + y
        return counts

    
    def create_dataframe_counts_specificGenH_fig(self, df, x, y):
        """Creates a DataFrame from the counts and percentages of two columns.
          
        Args:
            df: The original DataFrame.
            x: General health
            y: type
            z: Diabetes_binary
        Returns:
            A Pandas DataFrame containing counts, percentages, and the combined column.
        """
        filter1  = df['GeneralHealth'] == x
        filter2 = df['Type'] == y
        if y == "nondiabetic":
          z = 0
        else:
          z = 1
        
        filter3 = df['Diabetes_binary'] == z
        
        test = df [filter1 & filter2 & filter3] #
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
        cafe_colors =  ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',
                        'rgb(175, 51, 21)', 'rgb(35, 36, 21)']
        
        # Create subplots, using 'domain' type for pie charts
        specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
        fig = make_subplots(rows=2, cols=2, specs=specs)
        
        # Define pie charts
        fig.add_trace(go.Pie(labels=counts_gender['Gender'], values=counts_gender['percentage'], name='Gender',
                             marker_colors=night_colors), 1, 1)
        fig.add_trace(go.Pie(labels=counts_income['income'], values=counts_income['percentage'], name='Income',
                             marker_colors=sunflowers_colors), 1, 2)
        fig.add_trace(go.Pie(labels=counts_education['education'], values=counts_education['percentage'], name='Education',
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
        percentage   = (counts['count'] / len(df)) * 100
        counts["percentage"] = percentage.apply("{:.1f}%".format)  # Format as percentages
        counts['combined'] = counts[x] + ' and ' + counts[y]
        return counts
    
     
    def group_data(self, df,x,y):
        """
        Run all cleaning and transformation steps
    
        Returns
        -------
        counts and percentages of x,y in a DataFrames
        """

        # Assuming your DataFrame 'people' has columns 'GeneralHealth' and 'Type'

        # Filter data for relevant columns
        df = df[[x,y]]

        # Calculate counts using value_counts()
        counts = df[x].value_counts().to_frame(name="Count")

        # Calculate percentages (optional)
        #I want percentages:
        percentages = (counts["Count"] / len(df)) * 100
        counts["Percentage"] = percentages.apply("{:.1f}%".format)  # Format as percentages

        # Display the table
        return counts
        
    def cdc_bar_plot_combined_pie(self,df,x,y,t):
    
        """  test
        Dataframe, x, y axis
        Returns
        -------
        graph
        """ 
        summary =  self.create_dataframe_from_counts_part2(df, x,y)
        
        fig = go.Figure()
        
        # Prepare data with combined values for both labels and values
        data = [go.Pie(labels=summary['combined'], values=summary['percentage'])]
        
        # Add the trace to the figure
        fig.add_trace(*data)
        
        fig.update_layout(title='Distribution of Health by Type')                 
                       
        return fig