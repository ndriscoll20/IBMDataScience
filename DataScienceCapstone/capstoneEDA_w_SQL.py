# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:48:17 2022

@author: 1109336
"""
import csv, sqlite3
import pandas as pd

con = sqlite3.connect('my_data.db')
cur = con.cursor()

df = pd.read_csv('Spacex.csv')
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method='multi')

#Display the names of the unique launch sites  in the space mission
query1 = 'SELECT DISTINCT(Launch_Site) FROM SPACEXTBL;'

#Display 5 records where launch sites begin with the string 'CCA'
query2 =  'SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' limit 5;'

#Display the total payload mass carried by boosters launched by NASA (CRS)
query3 =  'SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE Customer = 'NASA (CRS)';'

#Display average payload mass carried by booster version F9 v1.1
query4 = 'SELECT PAYLOAD_MASS__KG_, Booster_Version FROM SPACEXTBL WHERE Booster_Version like 'F9 v1.1%';'
query4_2 = 'SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTBL WHERE Booster_Version like 'F9 v1.1%';'

#List the date when the first succesful landing outcome in ground pad was acheived.
query5 = 'SELECT DISTINCT("Landing _Outcome") FROM SPACEXTBL;'
query5_1 = 'SELECT min(Date) FROM SPACEXTBL WHERE "Landing _Outcome" like 'Success%';'

#List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
query6 = 'SELECT Booster_Version, "Landing _Outcome", PAYLOAD_MASS__KG_ FROM SPACEXTBL WHERE "Landing _Outcome" = "Success (drone ship)" AND PAYLOAD_MASS__KG_ >4000 AND PAYLOAD_MASS__KG_ < 6000;'

#List the total number of successful and failure mission outcomes
query7 = 'SELECT DISTINCT(Mission_Outcome) FROM SPACEXTBL;'
query7_1 = 'SELECT count(*) as Total, sum(case when Mission_Outcome like 'Success%' then 1 else 0 end) as Success, sum(case when Mission_Outcome like 'Failure%' then 1 else 0 end) as Failure FROM SPACEXTBL;'

#List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
query8 = 'SELECT Booster_Version, PAYLOAD_MASS__KG_ FROM SPACEXTBL WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL);''

#List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.
query9 = 'SELECT substr(Date,4,2), "Landing _Outcome", Booster_Version, Launch_Site FROM SPACEXTBL WHERE "Landing _Outcome" like '%Failure%' AND substr(Date,7,4) = '2015';'

#Rank the count of successful landing_outcomes between the date 04-06-2010 and 20-03-2017 in descending order.
query10 = 'SELECT * FROM SPACEXTBL WHERE Date > '04-06-2010' AND Date <'20-03-2017' AND "Landing _Outcome" like '%Success%' ORDER BY Date DESC;'