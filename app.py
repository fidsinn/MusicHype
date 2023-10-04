# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import joblib

import main

app = Flask(__name__)

df_all = pd.read_csv('data/df_all.csv')
X = pd.read_csv('data/X.csv')
X_train = pd.read_csv('data/X_train.csv')
X_train_knn = pd.read_csv('data/X_train_knn.csv')
sp_release = pd.read_csv('data/sp_release.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    slider_values = {
        # 'release_year': request.form['release_year'],
        'acousticness': request.form['acousticness'],
        'danceability': request.form['danceability'],
        'duration_sec': request.form['duration_sec'],
        'energy': request.form['energy'],
        'instrumentalness': request.form['instrumentalness'],
        'liveness': request.form['liveness'],
        'loudness': request.form['loudness'],
        'speechiness': request.form['speechiness'],
        'tempo': request.form['tempo'],
        'valence': request.form['valence'],
    }
    # slider_values = {
    #     # 'release_year': [2022],
    #     'acousticness': [0.00453],
    #     'danceability': [0.545],
    #     'duration_sec': [215],
    #     'energy': [0.641],
    #     'instrumentalness': [0.000066],
    #     'liveness': [0.171],
    #     'loudness': [-6.3],
    #     'speechiness': [0.0998],
    #     # 'time_signature',
    #     'tempo': [122],
    #     'valence': [0.464]
    # }

    # slider_values_list = list(slider_values.values())
    model_load = 'xgboost'  # 'xgboost' or 'regression'
    model = main.model_loader(model_load)
    # scaler = pickle.load(open('models/minmaxscaler.sav', 'rb'))
    # scaler = joblib.load('models/minmax_scaler.pkl')
    pop_pred, pop_place, sample_df_scaled = main.model_predictor(
        model_load=model_load, df_all=df_all, sample_data=slider_values, model=model)
    pop_dict = {
        'pop_pred': pop_pred,
        'pop_place': pop_place
    }
    df_neighbors_info = main.knn_model_loader(
        X=X, X_train=X_train, X_train_knn=X_train_knn, sp_release=sp_release, sample_df_scaled=sample_df_scaled)

    return render_template('result.html', slider_values=slider_values, pop_dict=pop_dict, df_neighbors_info=df_neighbors_info
                           )


if __name__ == '__main__':
    app.run(debug=True)
