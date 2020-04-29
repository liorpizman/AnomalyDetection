#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Script for tuning the window size parameter
'''
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from models.data_preprocessing.data_normalization import normalize_data
from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model
from models.time_series_model.time_series_estimator import TimeSeriesRegressor, time_series_split
from models.time_series_model.utils import safe_shape, mse, multi_mse
from utils.helper_methods import create_directories, get_current_time, get_training_data_lstm, anomaly_score_multi


def window_size_parameter_tuning_sklearn(ml_model, train_path, input_features, target_features, scaler, max_look_back):
    """
    Tune window size parameter over a constant range to get the optimal window size
    :param ml_model: machine learning model
    :param train_path: train set path
    :param input_features: input features list
    :param target_features: target features list
    :param scaler: scaler string
    :param max_look_back: maximum window size
    :return: MSEs test
    """

    df = pd.read_csv(train_path)

    original_input_df = df[input_features]

    input_df, X_train_scaler = normalize_data(data=original_input_df,
                                              scaler=scaler)

    original_target_df = df[target_features]

    target_df, Y_train_scaler = normalize_data(data=original_target_df,
                                               scaler=scaler)

    x_train, x_test, y_train, y_test = train_test_split(input_df, target_df, shuffle=False)

    n_prevs = range(1, max_look_back)
    mses_test = np.empty((len(n_prevs), safe_shape(y_test, 1)))

    for i, n_prev in enumerate(n_prevs):
        tsr = TimeSeriesRegressor(ml_model, n_prev=n_prev)
        tsr.fit(x_train, y_train)
        # get the (i+1)-th row
        mses_test[i, :] = mse(tsr.predict(x_test), y_test[n_prev:])

    return mses_test


def window_size_parameter_tuning_keras(train_path, input_features, target_features, scaler, max_look_back):
    """
    Tune window size parameter over a constant range to get the optimal window size - lstm
    :param train_path: train set path
    :param input_features: input features list
    :param target_features: target features list
    :param scaler: scaler string
    :param max_look_back: maximum window size
    :return: MSEs test
    """

    df = pd.read_csv(train_path)

    original_input_df = df[input_features]

    input_df, X_train_scaler = normalize_data(data=original_input_df,
                                              scaler=scaler)

    original_target_df = df[target_features]

    target_df, Y_train_scaler = normalize_data(data=original_target_df,
                                               scaler=scaler)

    x_train, x_test, y_train, y_test = train_test_split(input_df, target_df, shuffle=False)

    n_prevs = range(1, max_look_back)
    mses_test = np.empty((len(n_prevs), safe_shape(y_test, 1)))

    for i, n_prev in enumerate(n_prevs):
        X_train_preprocessed = get_training_data_lstm(x_train, n_prev)
        Y_train_preprocessed = get_training_data_lstm(y_train, n_prev)
        X_test_preprocessed = get_training_data_lstm(x_test, n_prev)
        y_test_preprocessed = get_training_data_lstm(y_test, n_prev)

        tsr = get_lstm_autoencoder_model(timesteps=n_prev,
                                         input_features=original_input_df.shape[1],
                                         target_features=original_target_df.shape[1],
                                         encoding_dimension=10,
                                         activation='relu',
                                         loss='mse',
                                         optimizer='adam')

        tsr.fit(X_train_preprocessed, Y_train_preprocessed, epochs=10, verbose=0)

        lstm = get_lstm_autoencoder_model(timesteps=n_prev,
                                          input_features=original_input_df.shape[1],
                                          target_features=original_target_df.shape[1],
                                          encoding_dimension=10,
                                          activation='relu',
                                          loss='mse',
                                          optimizer='adam')

        lstm.fit(X_train_preprocessed, Y_train_preprocessed, epochs=10, verbose=0)

        predicted = tsr.predict(X_test_preprocessed)
        lstm_predicted = lstm.predict(X_test_preprocessed)

        assert predicted == lstm_predicted

        actual = y_test_preprocessed

        assert predicted.shape == actual.shape

        mses_test[i, :] = multi_mse(predicted, y_test_preprocessed)

    return mses_test


def plot_tuning_results(mses_train, ml_name, input_path, scaler, factor):
    """
    Plot results of tuning window size parameter
    :param mses_train: numpy array of train MSEs
    :param ml_name: machine learning model name
    :param input_path: tuning directory
    :param scaler: scaler string
    :param factor: stable number to factor the matrix
    :return: plot displayed
    """

    c2 = "blue"
    c1 = "lightblue"
    p_color = "black"
    box_plot = plt.boxplot(np.transpose(mses_train) * factor, patch_artist=True)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[item], color=p_color)
    plt.setp(box_plot["boxes"], facecolor=c1)
    plt.setp(box_plot["fliers"], markeredgecolor=c2)
    # plt.boxplot(np.log(np.transpose(mses_train)))
    # plt.yscale('log')
    plt.title("Anomaly prediction over the simulator data set - {0} Model".format(ml_name))
    plt.ylabel("Testing log(MSE)s of the records = Actual MSE * {0}".format(factor))
    plt.xlabel("Setting of n_prev")

    min_value = 0.7 * np.amin(mses_train * factor)
    max_value = 1.3 * np.amax(mses_train * factor)

    plt.gcf().set_size_inches(12, 9)
    plt.gca().set_ylim([min_value, max_value])
    # plt.show()

    ml_directory_route = os.path.join(input_path, ml_name)
    create_directories(ml_directory_route)
    plot_directory_route = os.path.join(ml_directory_route, scaler)
    create_directories(plot_directory_route)
    current_time = get_current_time()
    plt.savefig(f'{plot_directory_route}/{ml_name}_{scaler}_{current_time}.png')
    plt.clf()


_input_features = [
    'Time',
    'Route Index',
    'GPS Distance',
    'Longitude',
    'Latitude',
    'Drone Speed',
    'Drone Climb',
    'Drone Altitude',
    'Accelerometer'
]

_target_features = [
    'RSSI0 OMNI',
    'RSSI1 OMNI',
    'CINR0 OMNI',
    'CINR1 OMNI',
    'Radio Distance',
    'Barometer Altitude'
]

# - Start - variables which should be changed

_train_path = 'C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\Scripts\\tuning\\without_anom.csv'
_tuning_path = 'C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\Scripts\\tuning'
_scaler = "min_max"

for i in range(1, 10):
    _ml_model = MultiOutputRegressor(SVR())
    _ml_name = "SVR"

    _max_look_back = 25
    _factor = 1

    _mses_train = window_size_parameter_tuning_sklearn(ml_model=_ml_model,
                                                       train_path=_train_path,
                                                       input_features=_input_features,
                                                       target_features=_target_features,
                                                       scaler=_scaler,
                                                       max_look_back=_max_look_back)

    plot_tuning_results(mses_train=_mses_train,
                        ml_name=_ml_name,
                        input_path=_tuning_path,
                        scaler=_scaler,
                        factor=_factor)

# for i in range(1, 10):
lstm_ml_name = "LSTM"
lstm_factor = 1
lstm_max_look_back = 25

# lstm_mses_train = window_size_parameter_tuning_keras(train_path=_train_path,
#                                                      input_features=_input_features,
#                                                      target_features=_target_features,
#                                                      scaler=_scaler,
#                                                      max_look_back=lstm_max_look_back)
#
# plot_tuning_results(mses_train=lstm_mses_train,
#                     ml_name=lstm_ml_name,
#                     input_path=_tuning_path,
#                     scaler=_scaler,
#                     factor=lstm_factor)
