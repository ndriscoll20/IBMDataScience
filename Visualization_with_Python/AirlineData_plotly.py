# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:02:44 2022

@author: Nick
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

airplanes = r'C:\Users\Nick\Documents\Python\Coursera\Visualization_with_Python\airline_data.csv'

airline_data =  pd.read_csv(airplanes,
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
#random sample:
data = airline_data.sample(n=500, random_state=42)

#Scatter Plot
fig = go.Figure(data=go.Scatter(x=data['Distance'], y=data['DepTime'],
                                mode ='markers', marker=dict(color='red')))
fig.update_layout(title='Distance vs Departure Time',
                  xaxis_title='Distance', yaxis_title='DepTime')
fig.show()

#Line Plot
line_data = data.groupby('Month')['ArrDelay'].mean().reset_index()
fig = go.Figure(data=go.Scatter(x=line_data['Month'],y=line_data['ArrDelay'],
                                mode='lines+markers', marker=dict(color='blue')))
fig.update_layout(title='Arrival Delay by Month of Year', 
                  xaxis_title='Month', yaxis_title='Delay Time')
fig.show()

#Bar Chart
bar_data = data.groupby(['DestState'])['Flights'].sum().reset_index()
fig = px.bar(bar_data, x="DestState", y="Flights", 
             title='Total number of flights to the destination state split by reporting airline') 
fig.show()

#Bubble Charts
bub_data = data.groupby('Reporting_Airline')['Flights'].sum().reset_index()
fig = px.scatter(bub_data, x='Reporting_Airline', y='Flights',
                 title='Number of Flights per Airline', 
                 size='Flights', hover_name='Reporting_Airline')
fig.show()

# Histogram 
data['ArrDelay'] = data['ArrDelay'].fillna(0)
fig = px.histogram(data, title='Distribution of Arrival Delays', x='ArrDelay')
fig.show()

#Pie Chart
fig = px.pie(data, values='Month',names='DistanceGroup',title='Distance group proportion by month')
fig.show()

#Sunburst Chart
fig = px.sunburst(data, path=['Month', 'DestStateName'], values='Flights')
fig.show()      
