#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
LSTM train and prediction execution function
'''
import pickle

import pandas as pd
import json

from models.lstm.lstm_hyper_parameters import lstm_hyper_parameters
from utils.constants import ATTACK_COLUMN
from utils.routes import *
from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model
from utils.helper_methods import get_training_data_lstm, get_testing_data_lstm, anomaly_score_multi, \
    get_threshold, report_results, get_method_scores, get_subdirectories, create_directories, get_current_time, plot, \
    plot_reconstruction_error_scatter, get_attack_boundaries

from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import MaxAbsScaler
from collections import defaultdict


def get_lstm_new_model_parameters():
    """
    Get LSTM hyper parameters
    :return: LSTM hyper parameters
    """

    return (
        lstm_hyper_parameters.get_window_size(),
        lstm_hyper_parameters.get_encoding_dimension(),
        lstm_hyper_parameters.get_activation(),
        lstm_hyper_parameters.get_loss(),
        lstm_hyper_parameters.get_optimizer(),
        lstm_hyper_parameters.get_threshold(),
        lstm_hyper_parameters.get_epochs()
    )


def run_model(training_data_path, test_data_path, results_path, similarity_score, save_model, new_model_running,
              algorithm_path, threshold, features_list, scalar_path):
    """
    Run LSTM model process
    :param training_data_path: train data set directory path
    :param test_data_path: test data set directory path
    :param results_path: results directory path
    :param similarity_score: chosen similarity functions
    :param save_model: indicator whether the user want to save the model or not
    :param new_model_running: indicator whether we are in new model creation flow or not
    :param algorithm_path: path of existing algorithm
    :param threshold: saved threshold for load model flow
    :param features_list:  saved chosen features for load model flow
    :param scalar_path: path of existing scalar directory
    :return: reported results for LSTM execution
    """

    # Choose between new model creation flow and load existing model flow
    if new_model_running:
        window_size, encoding_dimension, activation, loss, optimizer, threshold, epochs = get_lstm_new_model_parameters()
    else:
        lstm = load_model(algorithm_path)
        window_size = lstm.get_input_shape_at(0)[1]
        scalar = pickle.load(open(scalar_path, 'rb'))
        X_train = None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    create_directories(f'{results_path}/lstm/{current_time}')

    # Create sub directories for each similarity function
    for similarity in similarity_score:
        create_directories(f'{results_path}/lstm/{current_time}/{similarity}')

    # Train the model for each flight route
    for flight_route in FLIGHT_ROUTES:

        # Execute training for new model flow
        if new_model_running:
            lstm, scalar, X_train = execute_train(flight_route,
                                                  training_data_path=training_data_path,
                                                  results_path=f'{results_path}/lstm/{current_time}',
                                                  window_size=window_size,
                                                  encoding_dimension=encoding_dimension,
                                                  activation=activation,
                                                  loss=loss,
                                                  optimizer=optimizer,
                                                  add_plots=True,
                                                  features_list=features_list,
                                                  epochs=epochs)

        # Get results for each similarity function
        for similarity in similarity_score:
            current_results_path = f'{results_path}/lstm/{current_time}/{similarity}/{flight_route}'
            create_directories(current_results_path)
            tpr_scores, fpr_scores, delay_scores = execute_predict(flight_route,
                                                                   test_data_path=test_data_path,
                                                                   similarity_score=similarity,
                                                                   window_size=window_size,
                                                                   threshold=threshold,
                                                                   lstm=lstm,
                                                                   scalar=scalar,
                                                                   results_path=current_results_path,
                                                                   add_plots=True,
                                                                   run_new_model=new_model_running,
                                                                   X_train=X_train,
                                                                   features_list=features_list,
                                                                   save_model=save_model)

            df = pd.DataFrame(tpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_tpr.csv', index=False)

            df = pd.DataFrame(fpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_fpr.csv', index=False)

            df = pd.DataFrame(delay_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_delay.csv', index=False)

    algorithm_name = "LSTM"

    # Report results for training data to csv files
    for similarity in similarity_score:
        report_results(f'{results_path}/lstm/{current_time}/{similarity}',
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name)


def execute_train(flight_route,
                  training_data_path=None,
                  results_path=None,
                  window_size=None,
                  encoding_dimension=None,
                  activation=None,
                  loss=None,
                  optimizer=None,
                  add_plots=True,
                  features_list=None,
                  epochs=10):
    """
    Execute train function for a specific flight route
    :param flight_route: current flight route we should train on
    :param training_data_path: the path of training data directory
    :param results_path: the path of results directory
    :param window_size: window size variable
    :param encoding_dimension: encoding dimension variable
    :param activation: activation function
    :param loss: loss function
    :param optimizer: optimizer
    :param add_plots: indicator whether to add plots or not
    :param features_list: the list of features which the user chose
    :param epochs: num of epochs that was chosen by the user
    :return: LSTM model, normalization scalar, X_train data frame
    """

    df_train = pd.read_csv(f'{training_data_path}/{flight_route}/without_anom.csv')

    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    # Normalize the data
    X_train = scalar.fit_transform(df_train)
    X_train = get_training_data_lstm(X_train, window_size)

    # Get the model which is created by user's parameters
    lstm = get_lstm_autoencoder_model(window_size, df_train.shape[1],
                                      encoding_dimension, activation, loss, optimizer)
    history = lstm.fit(X_train, X_train, epochs=epochs, verbose=1).history

    # Add plots if the indicator is true
    if add_plots:
        plot(history['loss'], ylabel='loss', xlabel='epoch', title=f'{flight_route} Epoch Loss', plot_dir=results_path)

    return lstm, scalar, X_train


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    window_size=None,
                    threshold=None,
                    lstm=None,
                    scalar=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None,
                    save_model=False, ):
    """
    Execute predictions function for a specific flight route
    :param flight_route: current flight route we should train on
    :param test_data_path: the path of test data directory
    :param similarity_score: similarity function
    :param window_size: window size variable
    :param threshold: threshold from the train
    :param lstm: LSTM model
    :param scalar: normalization scalar
    :param results_path: the path of results directory
    :param add_plots: indicator whether to add plots or not
    :param run_new_model: indicator whether current flow is new model creation or not
    :param X_train: data frame
    :param features_list: the list of features which the user chose
    :param save_model: indicator whether the user want to save the model or not
    :return: tpr_scores, fpr_scores, delay_scores
    """

    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    delay_scores = defaultdict(list)

    # Set a threshold in new model creation flow
    if run_new_model:
        threshold = predict_train_set(lstm,
                                      X_train,
                                      save_model,
                                      add_plots,
                                      threshold,
                                      features_list,
                                      results_path,
                                      flight_route,
                                      similarity_score,
                                      scalar)

    flight_dir = os.path.join(test_data_path, flight_route)
    ATTACKS = get_subdirectories(flight_dir)

    # Iterate over all attacks in order to find anomalies
    for attack in ATTACKS:
        for flight_csv in os.listdir(f'{test_data_path}/{flight_route}/{attack}'):

            df_test_source = pd.read_csv(f'{test_data_path}/{flight_route}/{attack}/{flight_csv}')
            df_test_labels = df_test_source[[ATTACK_COLUMN]].values
            df_test = df_test_source[features_list]

            X_test = scalar.transform(df_test)
            X_test, y_test = get_testing_data_lstm(X_test, df_test_labels, window_size)

            X_pred = lstm.predict(X_test, verbose=1)
            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score_multi(X_test[i], pred, similarity_score))

            # Add plots if the indicator is true
            if add_plots:
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=y_test,
                                                  threshold=threshold,
                                                  plot_dir=results_path,
                                                  title=f'Outlier Score Testing for {flight_csv} in {flight_route}({attack})')

            predictions = [1 if x >= threshold else 0 for x in scores_test]

            attack_start, attack_end = get_attack_boundaries(df_test_source[ATTACK_COLUMN])

            method_scores = get_method_scores(predictions, run_new_model, attack_start, attack_end)

            tpr_scores[attack].append(method_scores[0])
            fpr_scores[attack].append(method_scores[1])
            delay_scores[attack].append(method_scores[2])

    return tpr_scores, fpr_scores, delay_scores


def predict_train_set(lstm,
                      X_train,
                      save_model,
                      add_plots,
                      threshold,
                      features_list,
                      results_path,
                      flight_route,
                      similarity_score,
                      scalar):
    """
    Execute prediction on the train data set
    :param lstm: LSTM model
    :param X_train: data frame
    :param save_model: indicator whether the user want to save the model or not
    :param add_plots: indicator whether to add plots or not
    :param threshold: threshold from the train
    :param features_list: the list of features which the user chose
    :param results_path: the path of results directory
    :param flight_route: current flight route we are working on
    :param similarity_score: similarity function
    :param scalar: normalization scalar
    :return: threshold
    """

    X_pred = lstm.predict(X_train, verbose=1)

    scores_train = []

    for i, pred in enumerate(X_pred):
        scores_train.append(anomaly_score_multi(X_train[i], pred, similarity_score))

    # choose threshold for which <LSTM_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
    threshold = get_threshold(scores_train, threshold)

    # Save created model if the indicator is true
    if save_model:
        data = {}
        data['features'] = features_list
        data['threshold'] = threshold
        with open(f'{results_path}/model_data.json', 'w') as outfile:
            json.dump(data, outfile)
        lstm.save(f'{results_path}/{flight_route}.h5')
        save_scalar_file_path = os.path.join(results_path, flight_route + "_scalar.pkl")
        with open(save_scalar_file_path, 'wb') as file:
            pickle.dump(scalar, file)

    # Add plots if the indicator is true
    if add_plots:
        plot_reconstruction_error_scatter(scores=scores_train, labels=[0] * len(scores_train), threshold=threshold,
                                          plot_dir=results_path,
                                          title=f'Outlier Score Training for {flight_route})')

    return threshold
