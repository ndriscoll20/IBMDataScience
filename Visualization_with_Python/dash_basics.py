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
import dash
import dash_html_components as html
import dash_core_components as dcc

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
# Add description about the graph using HTML P (paragraph) component
# Finally, add graph component.
app.layout = html.Div(children=[html.H1('Airline Dashboard',
                                        style={'textAlign': 'center', 
                                        'color': '#503D36', 
                                        'font-size': 40}),
                                html.P('Proportion of distance group (250 mile distance interval group) by flights.', 
                                        style={'textAlign':'center', 'color': '#F57241'}),
                                dcc.Graph(figure=fig),
                    ])

# Run the application                   
if __name__ == '__main__':
    app.run_server()