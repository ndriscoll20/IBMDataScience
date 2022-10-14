# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:06:33 2022

@author: Nick
"""

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Create a year column, remove parenthesis
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df = movies_df.drop('genres',axis=1)

ratings_df = ratings_df.drop('timestamp',axis=1)

userInput = [
    {'title': 'Breakfast Club, The','rating':5},
    {'title':'Toy Story','rating':3.5},
    {'title':'Jumanji','rating':2},
    {'title':'Pulp Fiction','rating':5},
    {'title':'Akira','rating':4.5}]

inputMovies = pd.DataFrame(userInput)

#Cross reference the user input with the movies list, and grab details from it
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year',axis=1)

userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
# Group by user since table rows are by movie, each with a unique userId
userSubsetGroup = userSubset.groupby(['userId'])
grp1130 = userSubsetGroup.get_group(1130)

# Sort so that users with movies most in common have priority
userSubsetGroup = sorted(userSubsetGroup, key=lambda x:len(x[1]), reverse=True)

userSubsetGroup = userSubsetGroup[0:100]

pearsonCorrelationDict = {}

# For every user group in the subset
for name, group in userSubsetGroup: 
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    nRatings = len(group) #n of summation in pearson eqn
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())] #Common Movies
    tempRatingList = temp_df['rating'].to_list() 
    tempGroupList = group['rating'].tolist() 
    
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    if Sxx !=0 and Syy !=0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
        
pearsonItems= pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

topUsers = pearsonDF.sort_values(by='similarityIndex',ascending=False)[0:50]
# Merge ratings df and pearson weight df
topUsersRating = topUsers.merge(ratings_df, left_on='userId',right_on='userId',how='inner')

topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

recommendation_df = pd.DataFrame()
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

top10 = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

