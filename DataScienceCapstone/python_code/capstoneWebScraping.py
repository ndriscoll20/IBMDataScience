# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:59:05 2022

@author: 1109336
"""
import sys
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd

wikiurl = 'https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches'

# HTML Scraping helper functions
'''
Each of the following function return the respective data (date/time, booster 
version,landing status, mass) from the HTML table cell, taking the element
of a table cell extract as input
'''
def date_time(table_cells):
    return [date_time.strip() for date_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    out  = [i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    mass = unicodedata.normalize('NFKD', table_cells.text).strip()
    if mass:
        mass.find('kg')
        new_mass=mass[0:mass.find('kg')+2]
    else: 
        new_mass = 0
    return new_mass

def extract_column_from_header(row):
    if (row.br):
        row.br.extract()
    if row.a: 
        row.a.extract()
    if row.sup:
        row.sup.extract()
    column_name = ' '.join(row.contents)

    if not(column_name.strip().isdigit()):
        column_name = column_name.strip()
        return column_name            
    
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

response = requests.get(static_url).text 
soup = BeautifulSoup(response, 'html5lib')
html_tables = soup.find_all('table')
first_launch_table = html_tables[2]

column_names = []

for header in html_tables[2].find_all('th'):
    if extract_column_from_header(header) is not None and len(extract_column_from_header(header)) > 0:
        column_names.append(extract_column_from_header(header))

launch_dict = dict.fromkeys(column_names)
del launch_dict['Date and time ( )']
#Initialize each as an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]

extracted_row = 0
#Extract each table 
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        #check to see if first table heading is as number corresponding to launch a number 
        if rows.th:
            if rows.th.string:
                flight_number=rows.th.string.strip()
                flag=flight_number.isdigit()
        else:
            flag=False
        #get table element 
        row=rows.find_all('td')
        #if it is number save cells in a dictonary 
        if flag:
            extracted_row += 1
            # Flight Number value
            #print(flight_number)
            launch_dict['Flight No.'].append(flight_number)
            datatimelist=date_time(row[0])
            
            # Date value
            date = datatimelist[0].strip(',')
            #print(date)
            launch_dict['Date'].append(date)
            
            # Time value
            time = datatimelist[1]
            #print(time)
            launch_dict['Time'].append(time)
              
            # Booster version
            bv=booster_version(row[1])
            if not(bv):
                bv=row[1].a.string
            #print(bv)
            launch_dict['Version Booster'].append(bv)
            
            # Launch Site
            launch_site = row[2].a.string
            #print(launch_site)
            launch_dict['Launch site'].append(launch_site)
            
            # Payload
            payload = row[3].a.string
            #print(payload)
            launch_dict['Payload'].append(payload)
            
            # Payload Mass
            payload_mass = get_mass(row[4])
            #print(payload)
            launch_dict['Payload mass'].append(payload)
            
            # Orbit
            orbit = row[5].a.string
            #print(orbit)
            launch_dict['Orbit'].append(orbit)
            
            # Customer
            if row[6].a is None:
                launch_dict['Customer'].append('Null')
            else: 
                customer = row[6].a.string
                #print(customer)
                launch_dict['Customer'].append(customer)
            
            # Launch outcome
            launch_outcome = list(row[7].strings)[0]
            #print(launch_outcome)
            launch_dict['Launch outcome'].append(launch_outcome)
            
            # Booster landing
            booster_landing = landing_status(row[8])
            #print(booster_landing)
            launch_dict['Booster landing'].append(booster_landing)
            
df = pd.DataFrame(launch_dict)
df.to_csv('spacex_web_scraped.csv', index=False)