import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import pickle

pd.set_option('display.max_columns', None)


def load_df_all():
    # audio features import
    audio_features = pd.read_csv('../data/audio_features.csv')

    # spotify import
    sp_artist_release = pd.read_csv('../data/sp_artist_release.csv')
    sp_artist_track = pd.read_csv('../data/sp_artist_track.csv')
    sp_artist = pd.read_csv('../data/sp_artist.csv')
    sp_release = pd.read_csv('../data/sp_release.csv')
    sp_track = pd.read_csv('../data/sp_track.csv')

    df_all = audio_features.merge(sp_track, on='isrc', how='inner')
    df_all = df_all.merge(sp_artist_track, on='track_id', how='inner')
    df_all = df_all.drop(columns=['updated_on_x'])
    df_all = df_all.merge(sp_artist, on='artist_id', how='inner')
    df_all = df_all.merge(sp_release, on='release_id', how='inner')

    df_all['duration_sec'] = df_all['duration_ms_x']/1000
    df_all['duration_sec'] = df_all['duration_sec'].round(0).astype(int)
    df_all = df_all.drop(columns=['duration_ms_x'])
    df_all = df_all[['isrc', 'release_id', 'track_title', 'artist_name', 'release_title', 'album_type', 'release_date', 'acousticness',
                     'danceability', 'duration_sec', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
                     'mode', 'speechiness', 'tempo', 'time_signature', 'valence', 'explicit', 'popularity']]

    df_all['key'] = df_all['key'].astype(str)
    df_all['mode'] = df_all['mode'].astype(str)
    df_all['time_signature'] = df_all['time_signature'].astype(str)
    df_all = df_all.drop(columns=['time_signature'])

    # df_all['release_year'] = df_all['release_date'].str.split('-').str[0].astype(int)
    df_all = df_all.drop(columns=['release_date'])

    pop_min = df_all['popularity'].min()
    pop_max = df_all['popularity'].max()
    df_all['popularity'] = round(
        (df_all['popularity']-pop_min)/(pop_max-pop_min)*100, 0)

    df_all = df_all[(df_all['duration_sec'] >= 60) &
                    (df_all['duration_sec'] <= 600)]

    # df_all['release_date'] = pd.to_datetime(df_all['release_date'], errors='coerce').dt.to_period('Y')

    df_all = df_all.groupby('isrc').agg({
        'release_id': 'first',
        'track_title': 'first',
        'artist_name': 'first',
        'release_title': 'first',
        'album_type': 'first',
        # 'release_year': 'first',
        'acousticness': 'first',
        'danceability': 'first',
        'duration_sec': 'first',
        'energy': 'first',
        'instrumentalness': 'first',
        # 'key': 'first',
        'liveness': 'first',
        'loudness': 'first',
        # 'mode': 'first',
        'speechiness': 'first',
        'tempo': 'first',
        # 'time_signature': 'first',
        'valence': 'first',
        # 'explicit': 'first',
        'popularity': 'mean'
    }).reset_index()

    df_all['popularity'] = df_all['popularity'].round(0).astype(int)

    # reduce lower popularity values to have a better distribution
    max_rows_per_value = 50000

    sampled_df = pd.DataFrame(columns=df_all.columns)

    for value in sorted(df_all['popularity'].unique()):
        subset = df_all[df_all['popularity'] == value]

        if len(subset) > max_rows_per_value:
            sampled_subset = subset.sample(
                n=max_rows_per_value, random_state=42)
        else:
            sampled_subset = subset

        sampled_df = pd.concat([sampled_df, sampled_subset])

    sampled_df = sampled_df.sample(
        frac=1, random_state=42).reset_index(drop=True)
    sampled_df.reset_index(drop=True, inplace=True)
    df_all = sampled_df

    return df_all


def scale_columns(df_all):
    columns_to_scale = [
        # 'release_year',
        'acousticness',
        'danceability',
        'duration_sec',
        'energy',
        'instrumentalness',
        'liveness',
        'loudness',
        'speechiness',
        'tempo',
        # 'time_signature',
        'valence'
    ]

    # STANDARD SCALER: scale numeric columns
    # scaler = StandardScaler()

    # MINMAX SCALER: scale numeric columns
    scaler = MinMaxScaler()

    df_all_scaled = df_all.copy()
    df_all_scaled[columns_to_scale] = scaler.fit_transform(
        df_all_scaled[columns_to_scale])

    return df_all_scaled


def train_test_split(df_all_scaled):
    X = df_all_scaled.drop(columns=['popularity'])
    # object_columns = X.select_dtypes(include=['object']).columns
    # X = pd.get_dummies(X, columns=object_columns)

    y = df_all_scaled['popularity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_train_knn = X_train.copy()
    X_train = X_train.drop(columns=[
                           'isrc', 'release_id', 'track_title', 'artist_name', 'release_title', 'album_type'])
    X_test = X_test.drop(columns=[
                         'isrc', 'release_id', 'track_title', 'artist_name', 'release_title', 'album_type'])
    return X_train, X_test, y_train, y_test, X, X_train_knn


def regression_model_builder(X_train, y_train):
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    pickle.dump(regression_model, open('models/regression_model.pkl', 'wb'))
    return regression_model


def xgboost_model_builder(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_tt = label_encoder.fit_transform(y_train)

    xgboost_model = xgb.XGBClassifier()
    xgboost_model.fit(X_train, y_train_tt)
    pickle.dump(xgboost_model, open('models/xgboost_model.pkl', 'wb'))
    return xgboost_model


def model_builder(model_load, X_train, y_train):
    if model_load == 'xgboost':
        model = xgboost_model_builder(X_train, y_train)
    elif model_load == 'regression':
        model = regression_model_builder(X_train, y_train)
    return model


def regression_model_loader():
    regression_model = pickle.load(open('models/regression_model.pkl', 'rb'))
    return regression_model


def xgboost_model_loader():
    xgboost_model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
    return xgboost_model


def model_loader(model_load):
    if model_load == 'xgboost':
        model = xgboost_model_loader()
    elif model_load == 'regression':
        model = regression_model_loader()
    return model


def regression_model_predictor(df_all, sample_data, regression_model):
    columns_to_scale = [
        # 'release_year',
        'acousticness',
        'danceability',
        'duration_sec',
        'energy',
        'instrumentalness',
        'liveness',
        'loudness',
        'speechiness',
        'tempo',
        # 'time_signature',
        'valence'
    ]

    sample_df = pd.DataFrame(sample_data, index=[0])

    scaler = MinMaxScaler()
    df_all_scaled = df_all.copy()

    df_all_scaled[columns_to_scale] = scaler.fit_transform(
        df_all_scaled[columns_to_scale])
    sample_df_scaled = scaler.transform(sample_df)

    predictions = regression_model.predict(sample_df_scaled)

    for i in predictions:
        i = i[0]
        pop_pred = int(round(i, 0))
        if pop_pred < 0:
            pop_pred = 1
        if pop_pred > 100:
            pop_pred = 99

        pop_under = df_all[df_all['popularity'] <= pop_pred].shape[0]
        pop_all = df_all.shape[0]

        pop_place = pop_under/pop_all*100
        pop_place = int(round(pop_place, 0))
        # print("Dein Song hat eine erwartete Popularität von %s und liegt über %s%% aller anderen Songs"%(pop_pred, pop_place))
    return pop_pred, pop_place, sample_df_scaled


def xgboost_model_predictor(df_all, sample_data, xgboost_model):
    columns_to_scale = [
        # 'release_year',
        'acousticness',
        'danceability',
        'duration_sec',
        'energy',
        'instrumentalness',
        'liveness',
        'loudness',
        'speechiness',
        'tempo',
        # 'time_signature',
        'valence'
    ]

    sample_df = pd.DataFrame(sample_data, index=[0])

    scaler = MinMaxScaler()
    df_all_scaled = df_all.copy()

    df_all_scaled[columns_to_scale] = scaler.fit_transform(
        df_all_scaled[columns_to_scale])
    sample_df_scaled = scaler.transform(sample_df)

    predictions = xgboost_model.predict(sample_df_scaled)

    for i in predictions:
        pop_pred = int(round(i, 0))
        if pop_pred < 0:
            pop_pred = 1
        if pop_pred > 100:
            pop_pred = 99

        pop_under = df_all[df_all['popularity'] <= pop_pred].shape[0]
        pop_all = df_all.shape[0]

        pop_place = pop_under/pop_all*100
        pop_place = int(round(pop_place, 0))
        # print("Dein Song hat eine erwartete Popularität von %s und liegt über %s%% aller anderen Songs"%(pop_pred, pop_place))
    return pop_pred, pop_place, sample_df_scaled


def model_predictor(model_load, df_all, sample_data, model):
    if model_load == 'xgboost':
        pop_pred, pop_place, sample_df_scaled = xgboost_model_predictor(
            df_all=df_all, sample_data=sample_data, xgboost_model=model)
    elif model_load == 'regression':
        pop_pred, pop_place, sample_df_scaled = regression_model_predictor(
            df_all=df_all, sample_data=sample_data, regression_model=model)
    return pop_pred, pop_place, sample_df_scaled


def knn_model_loader(X, X_train, X_train_knn, sp_release, sample_df_scaled):
    k = 6  # number of neighbors to find
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)

    distances, neighbor_indices = nn.kneighbors(sample_df_scaled)

    columns_sample_neighbors = [
        'isrc', 'track_title', 'artist_name', 'release_title']
    df_neighbors_info = pd.DataFrame(columns=columns_sample_neighbors)

    for i in list(X_train.iloc[neighbor_indices[0]].index):
        sample_neighbors = X_train.iloc[neighbor_indices[0]]
        sample_neighbors = sample_neighbors.reset_index()
        X_merger = X.reset_index()
        sample_neighbors = sample_neighbors.merge(
            X_merger, on='index', how='inner')
        sample_neighbors = sample_neighbors[[
            'index', 'isrc', 'track_title', 'artist_name', 'release_title']]
        sample_neighbors = sample_neighbors.merge(
            sp_release, on='release_title', how='inner')
        df_neighbors_info_append = X_train_knn.loc[i][[
            'isrc', 'track_title', 'artist_name', 'release_title']]
        df_neighbors_info_append = pd.DataFrame(df_neighbors_info_append).T
        # df_neighbors_info = df_neighbors_info.append(X_train_knn.loc[i][['isrc', 'track_title', 'artist_name','release_title']], ignore_index=True)
        df_neighbors_info = pd.concat(
            [df_neighbors_info, df_neighbors_info_append], axis=0)

    df_neighbors_info = df_neighbors_info.sample(frac=1)
    df_neighbors_info = df_neighbors_info.head(3)

    return df_neighbors_info
