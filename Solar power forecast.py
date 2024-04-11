#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb 
import pickle


# # Importing test weather data

# In[34]:


df_raw_test_data = pd.read_csv('df_weather_2018_2019.csv')
df_power = pd.read_csv('df_power_hourly_summer_2018.csv')
df_power = df_power.drop(df_power.index[0])


# In[35]:


#Merge data
df_test = pd.concat([df_power,df_raw_test_data], axis=1) 

#drop umnecesary columns
df_test = df_test.drop (columns = ['GHI [W/m²]', 'DNI [W/m²]', 'cloud_l [%]', 'cloud_m [%]', 'cloud_h [%]', 'is_day', 'date',
                                   'cloud [%]','precipitation [mm]', 'rain [mm]', 'snowfall [cm]', 'wind [km/h]'])
#set date to index
df_test['Date'] = pd.to_datetime(df_test['Date'], format='%d-%m-%Y %H:%M')
df_test = df_test.set_index('Date', drop=True)
df_test = df_test.dropna()

time_diff = (df_test.index.to_series().diff().dt.total_seconds() / 3600)
df_test['solar_power-1'] = df_test['solar_power_kW'].shift(1)
df_test.loc[time_diff > 2, 'solar_power-1'] = pd.NA
df_test = df_test.dropna()
df_test=df_test.iloc[:, [0,4,2,3,1]]
df_test


# Visualizing if there is a gap in the data for the chosen features. The plot and the table above show missing data after 17-08-2018 21:00

# In[36]:


import matplotlib.pyplot as plt

# Plot Temperature
plt.figure(figsize=(10, 5))

plt.plot(df_test.index, df_test['solar_power-1']/10, label='solar_power-1')
plt.plot(df_test.index, df_test['DR [W/m²]'], label='Direct Radiation [W/m²]')
plt.plot(df_test.index, df_test['Temp [C]']*10, label='Temperature [C]')


# Add title and labels
plt.title('Chosen features')
plt.xlabel('Time')
plt.legend()  # Add legend

# Show the plot
plt.show()


# # Test models

# In[37]:


with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)
    
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)


# In[38]:


# recurrent
Z=df_test.values
Y=Z[:,0]
X2=Z[:,[1,2,3,4]]


# In[39]:


RF_model


# In[40]:


GB_model


# The total registered solar power of Texel was 4391 kW in 2017 and 6161 kW in 2018. This is an increase which should be included into the model.

# In[41]:


sp_increase = 6161/4391


# In[42]:


rf_preds = RF_model.predict(X2)* sp_increase
gb_preds = GB_model.predict(X2)* sp_increase

# Blend the predictions
blended_preds = ((rf_preds + gb_preds) / 2)

# Apply bottom limit of zero to the blended predictions
blended_preds = np.maximum(blended_preds, 0)


# In[43]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Gradient Boosting Metrics
gb_mae = mean_absolute_error(Y, gb_preds)
gb_mse = mean_squared_error(Y, gb_preds)
gb_rmse = np.sqrt(gb_mse)
gb_mbe = np.mean(gb_preds - Y)
gb_nmbe = gb_mbe / np.mean(Y)  # Normalized Mean Bias Error

# Random Forest Metrics
rf_mae = mean_absolute_error(Y, rf_preds)
rf_mse = mean_squared_error(Y, rf_preds)
rf_rmse = np.sqrt(rf_mse)
rf_mbe = np.mean(rf_preds - Y)
rf_nmbe = rf_mbe / np.mean(Y)  # Normalized Mean Bias Error

# Blended Metrics
blended_mae = mean_absolute_error(Y, blended_preds)
blended_mse = mean_squared_error(Y, blended_preds)
blended_rmse = np.sqrt(blended_mse)
blended_mbe = np.mean(blended_preds - Y)
blended_nmbe = blended_mbe / np.mean(Y)  # Normalized Mean Bias Error

# Gradient Boosting cvMSE
gb_cv_mse = -cross_val_score(GB_model, X2, Y, cv=5, scoring='neg_mean_squared_error').mean()

# Random Forest cvMSE
rf_cv_mse = -cross_val_score(RF_model, X2, Y, cv=5, scoring='neg_mean_squared_error').mean()

# Blended cvMSE
blended_cv_mse = -cross_val_score(GB_model, X2, Y, cv=5, scoring='neg_mean_squared_error').mean()


# # Plots RF and GB models combined

# In[44]:


# Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(Y, gb_preds, color='blue', label='Gradiant boosting')
plt.scatter(Y, rf_preds, color='green', label='Random forest')
plt.scatter(Y, blended_preds, color='yellow', label='Combined')

plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, color='red', label='Ideal Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.legend()
plt.show()

# Create line plot of actual vs predicted values
plt.figure(figsize=(8, 6))

plt.plot(Y, color='blue', label='Actual Values')
plt.plot(gb_preds, color='red', label='Gradiant boosting')
plt.plot(rf_preds, color='green', label='Random forest')
plt.plot(blended_preds, color='yellow', label='Combined')

plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Line Plot of Actual vs Predicted Values')
plt.legend()
plt.show()


# # Preparing the dashboard

# In[ ]:


import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import matplotlib.pyplot as plt


# In[46]:


#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# In[47]:


# Define the values for each method
gb_metrics = [gb_mae, gb_mbe, gb_mse, gb_rmse, gb_cv_mse, gb_nmbe]
rf_metrics = [rf_mae, rf_mbe, rf_mse, rf_rmse, rf_cv_mse, rf_nmbe]
blended_metrics = [blended_mae, blended_mbe, blended_mse, blended_rmse, blended_cv_mse, blended_nmbe]

# Create the dictionary for df_metrics
metrics_data = {
    'Methods': ['Gradient Boosting', 'Random Forest', 'Combined'],
    'MAE': gb_metrics[:1] + rf_metrics[:1] + blended_metrics[:1],
    'MBE': gb_metrics[1:2] + rf_metrics[1:2] + blended_metrics[1:2],
    'MSE': gb_metrics[2:3] + rf_metrics[2:3] + blended_metrics[2:3],
    'RMSE': gb_metrics[3:4] + rf_metrics[3:4] + blended_metrics[3:4],
    'cvMSE': gb_metrics[4:5] + rf_metrics[4:5] + blended_metrics[4:5],
    'NMBE': gb_metrics[5:] + rf_metrics[5:] + blended_metrics[5:]
}

# Create df_metrics DataFrame
df_metrics = pd.DataFrame(data=metrics_data)

# Create the dictionary for df_forecast
forecast_data = {
    'Date': df_test.index.values,
    'Gradient Boosting': gb_preds,
    'Random Forest': rf_preds,
    'Combined': blended_preds
}

# Create df_forecast DataFrame
df_forecast = pd.DataFrame(data=forecast_data)


# In[48]:


df_forecast


# In[49]:


df_solar_power=df_test.drop(columns=['solar_power-1', 'DR [W/m²]', 'sunshine_duration', 'Temp [C]'])
df_results=pd.merge(df_solar_power,df_forecast, on='Date')

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:5])


# # Code for the dashboard

# In[59]:


import base64


# Load the raw data
df_raw = pd.read_csv('df_weather.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'], format='%d-%m-%Y %H:%M')  # Convert Date column to datetime

# Create the initial figure
fig1 = px.line(df_raw, x="date", y=df_raw.columns[1:14], title="Raw Data")

# Define the external CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Define column names for graphs and boxplots
columns_for_graphs = df_raw.columns[1:14]

# Create the Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Texel solar power fore cast'),
    html.P('Representing raw Data, Forecasting, and EDA'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-3'),  # Added third tab for EDA
    ]),
    html.Div(id='tabs-content')
])

# Define the callback to update the content based on the selected tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Weather data Texel'),
            dcc.Graph(
                id='yearly-data-raw',
                figure=fig1
            ),
            html.Div([
                dcc.Checklist(
                    id='column-selectors-raw',
                    options=[{'label': col, 'value': col} for col in df_raw.columns[1:14]],
                    value=df_raw.columns[1:14],  # Initial value: select all columns
                    labelStyle={'display': 'inline-block'}
                )
            ])
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Solar power Texel Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2
            ),
            generate_table(df_metrics)  # Call to generate_table function
        ])
    elif tab == 'tab-3':  # Added content for the third tab (EDA)
        return html.Div([
            html.H4('Exploratory Data Analysis'),
            dcc.Dropdown(
                id='column-selector',
                options=[{'label': col, 'value': col} for col in columns_for_graphs],
                value=columns_for_graphs[0],  # Initial value: first column
                style={'width': '50%'}
            ),
            html.Div(id='graph-container')
        ])

# Define the callback to update the boxplots and graphs based on the selected columns
@app.callback(
    Output('graph-container', 'children'),
    [Input('column-selector', 'value')]
)
def update_plots(column):
    # Update graph
    graph_children = None
    if column:
        if column in df_raw.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_raw[column])
            ax.set_title(column, fontsize=16)
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            plt.tight_layout()
            plt.savefig('graph.png')
            encoded_image = get_encoded_image('graph.png')
            graph_children = html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                dcc.Graph(
                    id='graph',
                    figure=px.box(df_raw, y=column, title=f'Boxplot of {column}')
                )
            ])
        else:
            graph_children = html.Div('Invalid column selected.')
    else:
        graph_children = html.Div('No column selected.')

    return graph_children

# Define the callback to update the raw data graph based on selected columns
@app.callback(
    Output('yearly-data-raw', 'figure'),
    [Input('column-selectors-raw', 'value')]
)
def update_graph(selected_columns):
    fig = px.line(df_raw, x="date", y=selected_columns, title="Raw Data")
    return fig

# Define the generate_table function
def generate_table(dataframe):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )

def get_encoded_image(image_path):
    with open(image_path, 'rb') as file:
        encoded_image = base64.b64encode(file.read()).decode('utf-8')
    return encoded_image


# Maybe you have to change the port number to a different one to run it.
if __name__ == '__main__':
    app.run_server(debug=False, port=8052)


# In[ ]:




