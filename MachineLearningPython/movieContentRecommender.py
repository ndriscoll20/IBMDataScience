# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:01:43 2022

@author: Nick
"""

import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

movies = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\movies.csv'
ratings = r'C:\Users\Nick\Documents\Python\Coursera\IBMDataScience\MachineLearningPython\ratings.csv'

movies_df = pd.read_csv(movies)
ratings_df = pd.read_csv(ratings)

# Create a year column, remove parenthesis
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')

moviesWithGenres_df = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
        
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

ratings_df = ratings_df.drop('timestamp',1)

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
inputMovies = inputMovies.drop('genres',axis=1).drop('year',axis=1)

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId',axis=1).drop('title',axis=1).drop('genres',axis=1).drop('year',axis=1)

#Create a weighted User profile
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId',axis=1).drop('title',axis=1).drop('genres',axis=1).drop('year',axis=1)

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

top20 = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]