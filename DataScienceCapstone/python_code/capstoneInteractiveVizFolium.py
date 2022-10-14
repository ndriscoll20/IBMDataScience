# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 09:37:42 2022

@author: Nick
"""

import folium
import pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from math import sin, cos, sqrt, atan2, radians


spacex_df = pd.read_csv(r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\Capstone\spacex_launch_geo.csv')

spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]

#Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))

marker = folium.map.Marker(
    nasa_coordinate, 
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0), 
        html = '<div style="font-size: 12; color:#d35400;">%s<b></div>' % 'NASA JSC',)
    )
site_map.add_child(circle)
site_map.add_child(marker)

# Add other sites to the map
point =[]
mark = []
for index, row in launch_sites_df.iterrows():
    site_name = row['Launch Site']
    site_coords = [row['Lat'],row['Long']]
   
    circle = folium.Circle(
        site_coords, radius=1000, 
        color='#d35400', 
        fill=True).add_child(folium.Popup(site_name))

    marker = (folium.map.Marker(
        site_coords, 
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0), 
            html = '<div style="font-size: 12; color:#d35400;">%s<b></div>' % site_name,)
        ))
    site_map.add_child(circle)
    site_map.add_child(marker)

marker_cluster = MarkerCluster()

def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

site_map.add_child(marker_cluster)

for index, record in spacex_df.iterrows():
    marker = folium.Marker([record['Lat'], record['Long']],
                           icon = folium.Icon(color='white', icon_color=record['marker_color']))
    marker_cluster.add_child(marker)

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

coast_coords = [28.56225, -80.56781] #Closest coast to CCAFS SLC-40
launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='CCAFS SLC-40',['Lat','Long']]
distance_coast = calculate_distance(launch_coords['Lat'],launch_coords['Long'], coast_coords[0], coast_coords[1])
distance_coast

# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon 
distance_marker = folium.Marker(
   coast_coords,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coast),
   )
)
site_map.add_child(distance_marker)

lines=folium.PolyLine(locations=[coast_coords, launch_coords], weight=1)
site_map.add_child(lines)

launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='KSC LC-39A',['Lat','Long']]
print(launch_coords)

# Create a marker with distance to a closest city, railway, highway, etc.
#Closest City
city_coords = [28.53748, -81.38123]
launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='KSC LC-39A',['Lat','Long']]
distance_city = calculate_distance(launch_coords['Lat'],launch_coords['Long'], city_coords[0], city_coords[1])

distance_marker = folium.Marker(
   city_coords,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_city),
   )
)
site_map.add_child(distance_marker)

lines=folium.PolyLine(locations=[city_coords, launch_coords], weight=1)
site_map.add_child(lines)

#Closest Railway
rail_coords = [34.6355, -120.62409]
launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='VAFB SLC-4E',['Lat','Long']]
distance_rail = calculate_distance(launch_coords['Lat'],launch_coords['Long'], rail_coords[0], rail_coords[1])

distance_marker = folium.Marker(
   rail_coords,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_rail),
   )
)
site_map.add_child(distance_marker)

lines=folium.PolyLine(locations=[rail_coords, launch_coords], weight=1)
site_map.add_child(lines)

#Check another city to see if closer
city_coords = [28.09864, -80.6369]
launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='KSC LC-39A',['Lat','Long']]
distance_city = calculate_distance(launch_coords['Lat'],launch_coords['Long'], city_coords[0], city_coords[1])

distance_marker = folium.Marker(
   city_coords,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_city),
   )
)
site_map.add_child(distance_marker)

lines=folium.PolyLine(locations=[city_coords, launch_coords], weight=1)
site_map.add_child(lines)

#Closest Highway
hway_coords = [28.52707, -80.78934]
launch_coords = launch_sites_df.loc[launch_sites_df['Launch Site']=='KSC LC-39A',['Lat','Long']]
distance_hway = calculate_distance(launch_coords['Lat'],launch_coords['Long'], hway_coords[0], hway_coords[1])

distance_marker = folium.Marker(
   hway_coords,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_hway),
   )
)
site_map.add_child(distance_marker)

lines=folium.PolyLine(locations=[hway_coords, launch_coords], weight=1)
site_map.add_child(lines)

site_map

site_map.save('site_map.html')