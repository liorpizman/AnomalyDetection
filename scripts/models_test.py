#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Run test scripts on different machine learning models
'''

import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model
from utils.helper_methods import get_training_data_lstm, multi_mean, plot_prediction_performance


def run_isolation_forest(file_path):
    """
    Run test for Isolation forest model
    :param file_path: the path of the data set
    :return: isolation forest prediction
    """

    features_list = ['Direction', 'Speed']
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')

    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    random_model = MultiOutputRegressor(
        RandomForestRegressor(max_depth=2, max_features="sqrt")
    )

    # lab_enc = preprocessing.LabelEncoder()
    # training_scores_encoded = lab_enc.fit_transform(X_train)
    random_model.fit(X_train, X_train)
    pred = random_model.predict(X_train)
    # isolation_model = MultiOutputRegressor(IsolationForest()).fit(X_train)
    # pred = isolation_model.predict(X_train)
    test_path = "C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\data_set\\test\\chicago_to_guadalajara\\down_attack"
    df_test = pd.read_csv(f'{test_path}/sensors_8.csv')
    df_test = df_test[features_list]

    Y_test = scalar.transform(df_test)
    test_pred = random_model.predict(Y_test)
    a = 4


def run_logistic_regression(file_path):
    """
    Run test for Logistic regression model
    :param file_path: the path of the data set
    :return: logistic regression prediction
    """

    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    logistic_model = LogisticRegression()

    # multi_model = MultiOutputRegressor(LogisticRegression())
    #
    # multi_model.fit(X_train, X_train)
    # multi_predict = multi_model.predict(X_train)

    logistic_model.fit(X_train, X_train)
    predict = logistic_model.predict(X_train)


def run_linear_regression(file_path):
    """
    Run test for Linear regression model
    :param file_path: the path of the data set
    :return: linear regression prediction
    """

    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ["Direction", "Speed", "Altitude", "lat", "long", "first_dis", "second_dis", "third_dis",
                     "fourth_dis"]
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    linear_model = LinearRegression()
    multi_model = MultiOutputRegressor(LinearRegression())

    linear_model.fit(X_train, X_train)
    multi_model.fit(X_train, X_train)

    linear_model_predict = linear_model.predict(X_train)
    multi_model_predict = multi_model.predict(X_train)

    print(linear_model_predict)
    print(multi_model_predict)


def run_MLP_model(file_path):
    """
    Run test for MLP model
    :param file_path: the path of the data set
    :return: MLP prediction
    """

    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    model = MLPRegressor()
    model.fit(X_train, X_train)
    pred = model.predict(X_train)

    multi_model = MultiOutputRegressor(MLPRegressor())
    multi_model.fit(X_train, X_train)
    multi_pred = model.predict(X_train)


def run_SGD(file_path):
    """
    Run test for Linear regression model
    :param file_path: the path of the data set
    :return: linear regression prediction
    """

    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    linear_model = SGDRegressor()
    multi_model = MultiOutputRegressor(SGDRegressor())

    multi_model.fit(X_train, X_train)

    multi_model_predict = multi_model.predict(X_train)

    print(multi_model_predict)


def run_lstm_performance_plot(file_path, result_path):
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Time', 'Route Index', 'GPS Distance', 'Longitude']

    target_features_list = ['CINR1 OMNI', 'Radio Distance', 'Barometer Altitude']

    input_df_train = df_train[features_list]
    target_df_train = df_train[target_features_list]

    window_size = 2

    # Step 1 : Clean train data set
    input_df_train = clean_data(input_df_train)

    target_df_train = clean_data(target_df_train)

    # Step 2: Normalize the data
    X_train, X_train_scaler = normalize_data(data=input_df_train,
                                             scaler="min_max")
    X_train_preprocessed = get_training_data_lstm(X_train, window_size)

    Y_train, Y_train_scaler = normalize_data(data=target_df_train,  # target data
                                             scaler="min_max")
    Y_train_preprocessed = get_training_data_lstm(Y_train, window_size)

    # Get the model which is created by user's parameters
    lstm = get_lstm_autoencoder_model(timesteps=window_size,
                                      input_features=input_df_train.shape[1],
                                      target_features=target_df_train.shape[1],
                                      encoding_dimension=8,
                                      activation='relu',
                                      loss='mean_squared_error',
                                      optimizer='Adam')
    history = lstm.fit(X_train_preprocessed, Y_train_preprocessed, epochs=5, verbose=0).history

    X_pred = lstm.predict(X_train_preprocessed, verbose=0)

    mean_y_train = multi_mean(Y_train_preprocessed)
    mean_x_pred = multi_mean(X_pred)

    assert mean_y_train.shape == mean_x_pred.shape

    for i, target_feature in enumerate(target_features_list):
        title = "Training performance of LSTM for " + target_feature
        plot_prediction_performance(Y_train=mean_y_train[:, i],
                                    X_pred=mean_x_pred[:, i],
                                    results_path=result_path,
                                    title=title,
                                    y_label="Sensor's Mean Value")


# path of the data set in the input
path = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\mini_set\\train\\Route_0"

result_path = "C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\mini_set\\results\\check"

run_lstm_performance_plot(path, result_path)
