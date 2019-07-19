# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:25:09 2019

@author: praful
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################
    
df = pd.read_csv("BollywoodMovieDetail.csv")
print(df.head())
print(df.columns)

v=df['title'].count()
print(v)
indexx=[]
for x in range(0,v):
    indexx.append(x)
    
df["index"]=indexx

features = ['title','genre','actors', 'directors']

for feature in features:
	df[feature] = df[feature].fillna('')

print(df.head())

def combine_features(row):
    return row["title"] +" "+row["genre"]+" "+row["actors"]+" "+row["directors"]

df["combined_features"] = df.apply(combine_features,axis=1)
#print(df["combined_features"][])

cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "Vaastu Shastra"
movie_index = get_index_from_title(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>50:
			break