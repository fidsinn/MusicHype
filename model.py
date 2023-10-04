import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import pickle

pd.set_option('display.max_columns', None)

#from flask import Flask, render_template, request, redirect, url_for, flash
#from flask_sqlalchemy import SQLAlchemy
#import requests

#audio features import
audio_features = pd.read_csv('data/audio_features.csv')

#spotify import
sp_artist_release = pd.read_csv('data/sp_artist_release.csv')
sp_artist_track = pd.read_csv('data/sp_artist_track.csv')
sp_artist = pd.read_csv('data/sp_artist.csv')
sp_release = pd.read_csv('data/sp_release.csv')
sp_track = pd.read_csv('data/sp_track.csv')

#preprocessing
df_all = audio_features.merge(sp_track, on='isrc', how='inner')
df_all = df_all.merge(sp_artist_track, on='track_id', how='inner')
df_all = df_all.drop(columns=['updated_on_x'])
df_all = df_all.merge(sp_artist, on='artist_id', how='inner')
df_all = df_all.merge(sp_release, on='release_id', how='inner')

df_all['duration_sec'] = df_all['duration_ms_x']/1000
df_all['duration_sec'] = df_all['duration_sec'].round(0).astype(int)
df_all = df_all.drop(columns=['duration_ms_x'])
df_all = df_all[['isrc', 'release_id', 'track_title', 'artist_name','release_title', 'album_type', 'release_date', 'acousticness',
                       'danceability', 'duration_sec', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
                       'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'explicit', 'popularity']]

df_all['key'] = df_all['key'].astype(str)
df_all['mode'] = df_all['mode'].astype(str)
df_all['time_signature'] = df_all['time_signature'].astype(str)
df_all = df_all.drop(columns=['time_signature'])

#df_all['release_year'] = df_all['release_date'].str.split('-').str[0].astype(int)
#df_all = df_all[df_all['release_year']>=2000]
df_all = df_all.drop(columns=['release_date'])

df_all = df_all[df_all['acousticness'].between(0.0, 0.2)]
df_all = df_all[df_all['danceability'].between(0.1, 1.0)]
df_all = df_all[df_all['duration_sec'].between(60, 600)]
df_all = df_all[df_all['energy'].between(0.0, 1.0)]
df_all = df_all[df_all['liveness'].between(0.0, 0.4)]
df_all = df_all[df_all['loudness'].between(-20.0, 0.0)]
df_all = df_all[df_all['speechiness'].between(0.02, 0.4)]
df_all = df_all[df_all['tempo'].between(70, 180)]
#df_all = df_all[df_all['tempo'].between(50, 200)]

pop_min = df_all['popularity'].min()
pop_max = df_all['popularity'].max()
df_all['popularity'] = round((df_all['popularity']-pop_min)/(pop_max-pop_min)*100, 0)

df_all = df_all[(df_all['duration_sec']>=60) & (df_all['duration_sec']<=600)]

#df_all['release_date'] = pd.to_datetime(df_all['release_date'], errors='coerce').dt.to_period('Y')

df_all = df_all.groupby('isrc').agg({
    'release_id': 'first',
    'track_title': 'first',
    'artist_name': 'first',
    'release_title': 'first',
    'album_type': 'first',
    #'release_year': 'first',
    'acousticness': 'first',
    'danceability': 'first',
    'duration_sec': 'first',
    'energy': 'first',
    'instrumentalness': 'first',
    #'key': 'first',
    'liveness': 'first',
    'loudness': 'first',
    #'mode': 'first',
    'speechiness': 'first',
    'tempo': 'first',
    #'time_signature': 'first',
    'valence': 'first',
    #'explicit': 'first',
    'popularity': 'mean'
}).reset_index()

df_all['popularity'] = df_all['popularity'].round(0).astype(int)

#register min and max values of all columns that are standardized later
#release year
#release_year_min = df_all['release_year'].min()
#release_year_max = df_all['release_year'].max()
#acousticness
acousticness_min = df_all['acousticness'].min()
acousticness_max = df_all['acousticness'].max()
#danceability
danceability_min = df_all['danceability'].min()
danceability_max = df_all['danceability'].max()
#duration_sec
duration_sec_min = df_all['duration_sec'].min()
duration_sec_max = df_all['duration_sec'].max()
#energy
energy_min = df_all['energy'].min()
energy_max = df_all['energy'].max()
#instrumentalness
instrumentalness_min = df_all['instrumentalness'].min()
instrumentalness_max = df_all['instrumentalness'].max()
#liveness
liveness_min = df_all['liveness'].min()
liveness_max = df_all['liveness'].max()
#loudness
loudness_min = df_all['loudness'].min()
loudness_max = df_all['loudness'].max()
#speechiness
speechiness_min = df_all['speechiness'].min()
speechiness_max = df_all['speechiness'].max()
#tempo
tempo_min = df_all['tempo'].min()
tempo_max = df_all['tempo'].max()
#valence
valence_min = df_all['valence'].min()
valence_max = df_all['valence'].max()
#popularity
popularity_min = df_all['popularity'].min()
popularity_max = df_all['popularity'].max()

# reduce lower popularity values to have a better distribution
max_rows_per_value = 100000

sampled_df = pd.DataFrame(columns=df_all.columns)

for value in sorted(df_all['popularity'].unique()):
    subset = df_all[df_all['popularity'] == value]
    
    if len(subset) > max_rows_per_value:
        sampled_subset = subset.sample(n=max_rows_per_value, random_state=42)
    else:
        sampled_subset = subset
    
    sampled_df = pd.concat([sampled_df, sampled_subset])

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
sampled_df.reset_index(drop=True, inplace=True)
df_all = sampled_df

df_all.to_csv('data/df_all.csv', index=False)
print('df_all_size:', df_all.shape)

# print all attributes
#print('release_year_min:', release_year_min)
#print('release_year_max:', release_year_max)
print('acousticness_min:', acousticness_min)
print('acousticness_max:', acousticness_max)
print('danceability_min:', danceability_min)
print('danceability_max:', danceability_max)
print('duration_sec_min:', duration_sec_min)
print('duration_sec_max:', duration_sec_max)
print('energy_min:', energy_min)
print('energy_max:', energy_max)
print('instrumentalness_min:', instrumentalness_min)
print('instrumentalness_max:', instrumentalness_max)
print('liveness_min:', liveness_min)
print('liveness_max:', liveness_max)
print('loudness_min:', loudness_min)
print('loudness_max:', loudness_max)
print('speechiness_min:', speechiness_min)
print('speechiness_max:', speechiness_max)
print('tempo_min:', tempo_min)
print('tempo_max:', tempo_max)
print('valence_min:', valence_min)
print('valence_max:', valence_max)
print('popularity_min:', popularity_min)
print('popularity_max:', popularity_max)

#
columns_to_scale = [
                    #'release_year',
                    'acousticness',
                    'danceability',
                    'duration_sec',
                    'energy',
                    'instrumentalness',
                    'liveness',
                    'loudness',
                    'speechiness',
                    'tempo',
                    #'time_signature',
                    'valence'
                    ]

#STANDARD SCALER: scale numeric columns
# scaler = StandardScaler()

#MINMAX SCALER: scale numeric columns
scaler = MinMaxScaler()

pickle.dump(scaler, open('models/minmaxscaler.sav', 'wb'))
joblib.dump(scaler, 'models/minmax_scaler.pkl')

df_all_scaled = df_all.copy()
df_all_scaled[columns_to_scale] = scaler.fit_transform(df_all_scaled[columns_to_scale])

#
X = df_all_scaled.drop(columns=['popularity'])
#object_columns = X.select_dtypes(include=['object']).columns
#X = pd.get_dummies(X, columns=object_columns)

y = df_all_scaled['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_knn = X_train.copy()
X_train = X_train.drop(columns=['isrc', 'release_id', 'track_title', 'artist_name','release_title', 'album_type'])
X_test = X_test.drop(columns=['isrc', 'release_id', 'track_title', 'artist_name','release_title', 'album_type'])

X.to_csv('data/X.csv', index=False)
X_train.to_csv('data/X_train.csv', index=False)

model_load = False

if model_load:
    regression_model = pickle.load(open('models/regression_model.sav', 'rb'))

else:
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    pickle.dump(regression_model, open('models/regression_model.sav', 'wb'))

y_pred = regression_model.predict(X_test)

pop_pred_min = y_pred.min()
pop_pred_max = y_pred.max()

y_pred = (y_pred-pop_pred_min)/(pop_pred_max-pop_pred_min)*100
y_pred = np.round(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Create a new DataFrame for the new data you want to make predictions on
sample_data = {
    #'release_year': [0.995, 0.90, 0.99],
    'acousticness': [1.0, 0.5, 0.053313],
    'danceability': [1.0, 0.5, 0.763000],
    'duration_sec': [0.43, 0.14, 0.28],
    'energy': [1.0, 0.5, 0.628000],
    'instrumentalness': [1.0, 0.5, 0.000000],
    'liveness':[1.0, 0.5, 0.114000],
    'loudness': [1.0, 0.5, 0.826907],
    'speechiness': [1.0, 0.5, 0.051653],
    #'time_signature',
    'tempo': [1.0, 0.5, 0.564000],
    'valence': [1.0, 0.5, 0.193000]
}

columns_to_scale = [
                    #'release_year',
                    'acousticness',
                    'danceability',
                    'duration_sec',
                    'energy',
                    'instrumentalness',
                    'liveness',
                    'loudness',
                    'speechiness',
                    'tempo',
                    #'time_signature',
                    'valence'
                    ]

sample_df = pd.DataFrame(sample_data)

predictions = regression_model.predict(sample_df)

for i in predictions:
    pop_under = df_all[df_all['popularity']<=i].shape[0]
    pop_all = df_all.shape[0]
    pop_place = pop_under/pop_all*100
    pop_place = int(round(pop_place, 0))
    i = int(round(i, 0))
    print("Dein Song hat eine erwartete Popularität von %s%% und liegt über %s%% aller anderen Songs"%(i, pop_place))

# Create a new DataFrame for the new data you want to make predictions on
sample_data = {
    #'release_year': [2000],
    'acousticness': [0.5],
    'danceability': [0.5],
    'duration_sec': [130],
    'energy': [0.5],
    'instrumentalness': [0.5],
    'liveness':[0.5],
    'loudness': [1],
    'speechiness': [0.5],
    #'time_signature',
    'tempo': [100],
    'valence': [0.5]
}

sample_data = {
    #'release_year': [2022],
    'acousticness': [0.00453],
    'danceability': [0.545],
    'duration_sec': [215],
    'energy': [0.641],
    'instrumentalness': [0.000066],
    'liveness':[0.171],
    'loudness': [-6.3],
    'speechiness': [0.0998],
    #'time_signature',
    'tempo': [122],
    'valence': [0.464]
}

columns_to_scale = [
                    #'release_year',
                    'acousticness',
                    'danceability',
                    'duration_sec',
                    'energy',
                    'instrumentalness',
                    'liveness',
                    'loudness',
                    'speechiness',
                    'tempo',
                    #'time_signature',
                    'valence'
                    ]

sample_df = pd.DataFrame(sample_data)

df_all_scaled = df_all.copy()
df_all_scaled[columns_to_scale] = scaler.fit_transform(df_all_scaled[columns_to_scale])

sample_df_scaled = scaler.transform(sample_df)

predictions = regression_model.predict(sample_df_scaled)

for i in predictions:
    pop_under = df_all[df_all['popularity']<=i].shape[0]
    pop_all = df_all.shape[0]
    pop_place = pop_under/pop_all*100
    pop_place = int(round(pop_place, 0))
    i = int(round(i, 0))
    print("Dein Song hat eine erwartete Popularität von %s und liegt über %s%% aller anderen Songs"%(i, pop_place))

#KNN

k = 3 # number of neighbors to find
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_train)

distances, neighbor_indices = nn.kneighbors(sample_df_scaled)

columns_sample_neighbors = ['isrc', 'track_title', 'artist_name','release_title']
df_neighbors_info = pd.DataFrame(columns=columns_sample_neighbors)

for i in list(X_train.iloc[neighbor_indices[0]].index):
    sample_neighbors = X_train.iloc[neighbor_indices[0]]
    sample_neighbors = sample_neighbors.reset_index()
    X_merger = X.reset_index()
    sample_neighbors = sample_neighbors.merge(X_merger, on='index', how='inner')
    sample_neighbors = sample_neighbors[['index', 'isrc', 'track_title', 'artist_name', 'release_title']]
    sample_neighbors = sample_neighbors.merge(sp_release, on='release_title', how='inner')
    print(X_train_knn.loc[i][['isrc', 'track_title', 'artist_name','release_title']])
    df_neighbors_info_append = X_train_knn.loc[i][['isrc', 'track_title', 'artist_name','release_title']]
    df_neighbors_info_append = pd.DataFrame(df_neighbors_info_append).T
    #df_neighbors_info = df_neighbors_info.append(X_train_knn.loc[i][['isrc', 'track_title', 'artist_name','release_title']], ignore_index=True)
    df_neighbors_info = pd.concat([df_neighbors_info, df_neighbors_info_append], axis=0)