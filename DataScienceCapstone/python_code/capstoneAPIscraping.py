# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 08:56:15 2022

@author: Nick
"""

import pandas as pd
import numpy as np
import requests
import datetime

pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth',None)

def getBoosterVersion(data):
    for x in data['rocket']:
         if x:
             response = requests.get('https://api.spacexdata.com/v4/rockets/'+str(x)).json()
             BoosterVersion.append(response['name'])
             
def getLaunchSite(data):
    for x in data['launchpad']:
         if x:
             response = requests.get('https://api.spacexdata.com/v4/launchpads/'+str(x)).json()
             Longitude.append(response['longitude'])
             Latitude.append(response['latitude'])
             LaunchSite.append(response['name'])

def getPayloadData(data):
    for load in data['payloads']:
         if load:
             response = requests.get('https://api.spacexdata.com/v4/payloads/'+str(load)).json()
             PayloadMass.append(response['mass_kg'])
             Orbit.append(response['orbit'])

def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

# spacex_url = 'https://api.spacexdata.com/v4/launches/past'
# response = requests.get(spacex_url)

#Use a static Json for project so that doesn't change
static_json_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
response = requests.get(static_json_url)
data = pd.json_normalize(response.json())

# Clean up some data
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]
# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]
# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

data['date'] = pd.to_datetime(data['date_utc']).dt.date
#Restrict the dates of launches to pre-11/13/2020
data = data[data['date'] <= datetime.date(2020, 11, 13)]

#Global variables 
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)

#Combine data into a dictionary
launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}

launch_df = pd.DataFrame(launch_dict)
#launch_df.head()
launch_df = launch_df[launch_df['BoosterVersion']!='Falcon 1']
#Reset the FlightNumber 
launch_df.loc[:'FlightNumber'] = list(range(1, launch_df.shape[0]+1))

launch_df.isnull().sum()
payloadMean = launch_df.PayloadMass.mean()
launch_df['PayloadMass'] = launch_df['PayloadMass'].replace(np.nan, payloadMean)

launch_df.to_csv('dataset_part_1.csv',index=False)