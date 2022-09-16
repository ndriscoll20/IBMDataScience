# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:06:22 2022

@author: Nick
"""
#pip3 install pandas dash
#pip3 install httpx==0.20 dash plotly
#python3 -m pip install pandas dash  

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

airplanes = r'C:\Users\Nick\Documents\Python\Coursera\Visualization_with_Python\airline_data.csv'

airline_data =  pd.read_csv(airplanes,
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

data = airline_data.sample(n=500, random_state=42)
fig = px.pie(data, values='Flights', names='DistanceGroup', title='Distance group proportion by flights')

# Create a dash application
app = dash.Dash(__name__)

# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 component
# Add a html.Div and core input text component
# Finally, add graph component.

app.layout = html.Div(children=[html.H1('Airline Performance Dashboard',
                                        style={'textAlign': 'center', 
                                        'color': '#503D36', 
                                        'font-size': 40}),
                                html.Div(["Input Year", dcc.Input(id='input-year', value='2010',
                                                               type='number', style={'height':'50px',
                                                                                    'font-size':35}),],
                                         style={'font-size':40}),
                                html.Br(),
                                html.Br(),
                                html.Div(dcc.Graph(id='line-plot')),
                                ])

@app.callback( Output(component_id='line-plot', component_property='figure'), 
               Input(component_id='input-year', component_property='value'))

def get_graph(entered_year):
    df = airline_data[airline_data['Year']==int(entered_year)]
    
    line_data = df.groupby('Month')['ArrDelay'].mean().reset_index()
    
    fig=go.Figure(data=go.Scatter(x=line_data['Month'],y=line_data['ArrDelay'], mode='lines',
                    marker=dict(color='green')))
    fig.update_layout(title='Month vs. Average Flight Delay Time',xaxis_title='Month', yaxis_title='ArrDelay')

    return fig

# Run the application                   
if __name__ == '__main__':
    app.run_server()
    
    