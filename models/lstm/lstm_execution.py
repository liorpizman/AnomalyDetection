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

from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.lstm.lstm_hyper_parameters import lstm_hyper_parameters
from utils.constants import ATTACK_COLUMN
from utils.routes import *
from models.lstm.lstm_autoencoder import get_lstm_autoencoder_model
from utils.helper_methods import get_training_data_lstm, get_testing_data_lstm, anomaly_score_multi, \
    get_threshold, report_results, get_method_scores, get_subdirectories, create_directories, get_current_time, plot, \
    plot_reconstruction_error_scatter, get_attack_boundaries
from tensorflow.python.keras.models import load_model
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
              algorithm_path, threshold, features_list, target_features_list, train_scaler_path, target_scaler_path):
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
    :param features_list:  the list of features which the user chose for the train
    :param target_features_list: all the features in the test data set for the target
    :param train_scaler_path: path of existing input train scaler directory
    :param target_scaler_path: path of existing input target scaler directory
    :return: reported results for LSTM execution
    """

    # Choose between new model creation flow and load existing model flow
    if new_model_running:
        window_size, encoding_dimension, activation, loss, optimizer, threshold, epochs = get_lstm_new_model_parameters()
    else:
        lstm = load_model(algorithm_path)
        window_size = lstm.get_input_shape_at(0)[1]
        X_train_scaler = pickle.load(open(train_scaler_path, 'rb'))
        Y_train_scaler = pickle.load(open(target_scaler_path, 'rb'))
        X_train = None
        Y_train = None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    current_time_path = os.path.join(*[str(results_path), 'lstm', str(current_time)])
    create_directories(f"{current_time_path}")

    # Create sub directories for each similarity function
    for similarity in similarity_score:
        similarity_path = os.path.join(*[str(current_time_path), str(similarity)])
        create_directories(f"{similarity_path}")

    # Train the model for each flight route
    for flight_route in FLIGHT_ROUTES:

        # Execute training for new model flow
        if new_model_running:
            lstm, X_train_scaler, Y_train_scaler, X_train, Y_train = execute_train(flight_route,
                                                                                   training_data_path=training_data_path,
                                                                                   results_path=f"{current_time_path}",
                                                                                   window_size=window_size,
                                                                                   encoding_dimension=encoding_dimension,
                                                                                   activation=activation,
                                                                                   loss=loss,
                                                                                   optimizer=optimizer,
                                                                                   add_plots=True,
                                                                                   features_list=features_list,
                                                                                   epochs=epochs,
                                                                                   target_features_list=target_features_list)

        # Get results for each similarity function
        for similarity in similarity_score:
            current_results_path = os.path.join(*[str(current_time_path), str(similarity), str(flight_route)])
            create_directories(f"{current_results_path}")
            tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration = execute_predict(flight_route,
                                                                                                test_data_path=test_data_path,
                                                                                                similarity_score=similarity,
                                                                                                window_size=window_size,
                                                                                                threshold=threshold,
                                                                                                lstm=lstm,
                                                                                                X_train_scaler=X_train_scaler,
                                                                                                results_path=current_results_path,
                                                                                                add_plots=True,
                                                                                                run_new_model=new_model_running,
                                                                                                X_train=X_train,
                                                                                                features_list=features_list,
                                                                                                target_features_list=target_features_list,
                                                                                                save_model=save_model,
                                                                                                Y_train_scaler=Y_train_scaler,
                                                                                                Y_train=Y_train)

            df = pd.DataFrame(tpr_scores)
            tpr_path = os.path.join(*[str(current_results_path), str(flight_route) + '_tpr.csv'])
            df.to_csv(f"{tpr_path}", index=False)

            df = pd.DataFrame(fpr_scores)
            fpr_path = os.path.join(*[str(current_results_path), str(flight_route) + '_fpr.csv'])
            df.to_csv(f"{fpr_path}", index=False)

            df = pd.DataFrame(acc_scores)
            acc_path = os.path.join(*[str(current_results_path), str(flight_route) + '_acc.csv'])
            df.to_csv(f"{acc_path}", index=False)

            df = pd.DataFrame(delay_scores)
            delay_path = os.path.join(*[str(current_results_path), str(flight_route) + '_delay.csv'])
            df.to_csv(f"{delay_path}", index=False)

    algorithm_name = "LSTM"

    # Report results for training data to csv files
    for similarity in similarity_score:
        report_similarity_path = os.path.join(*[str(results_path), 'lstm', str(current_time), str(similarity)])
        report_results(f"{report_similarity_path}",
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name,
                       similarity,
                       routes_duration)


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
                  epochs=10,
                  target_features_list=None):
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
    :param target_features_list: the list of features which the user chose for the target
    :return: LSTM model, normalization input train scalar,normalization input target scalar, X_train data frame,Y_train data frame
    """

    without_anomaly_path = os.path.join(*[str(training_data_path), str(flight_route), 'without_anom.csv'])
    df_train = pd.read_csv(f"{without_anomaly_path}")

    input_df_train = df_train[features_list]
    target_df_train = df_train[target_features_list]

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
                                      encoding_dimension=encoding_dimension,
                                      activation=activation,
                                      loss=loss,
                                      optimizer=optimizer)
    history = lstm.fit(X_train_preprocessed, Y_train_preprocessed, epochs=epochs, verbose=1).history

    # Add plots if the indicator is true
    if add_plots:
        plot(history['loss'], ylabel='loss', xlabel='epoch', title=f'{flight_route} Epoch Loss', plot_dir=results_path)

    return lstm, X_train_scaler, Y_train_scaler, X_train_preprocessed, Y_train_preprocessed


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    window_size=None,
                    threshold=None,
                    lstm=None,
                    X_train_scaler=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None,
                    target_features_list=None,
                    save_model=False,
                    Y_train_scaler=None,
                    Y_train=None):
    """
    Execute predictions function for a specific flight route
    :param flight_route: current flight route we should train on
    :param test_data_path: the path of test data directory
    :param similarity_score: similarity function
    :param window_size: window size variable
    :param threshold: threshold from the train
    :param lstm: LSTM model
    :param X_train_scaler: normalization train input scalar
    :param results_path: the path of results directory
    :param add_plots: indicator whether to add plots or not
    :param run_new_model: indicator whether current flow is new model creation or not
    :param X_train: train input data frame
    :param features_list: the list of features which the user chose for the input
    :param target_features_list: the list of features which the user chose for the target
    :param save_model: indicator whether the user want to save the model or not
    :param Y_train_scaler: normalization train target scalar
    :param Y_train: train target data frame
    :return: tpr scores, fpr scores, acc scores, delay scores, routes duration
    """

    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    acc_scores = defaultdict(list)
    delay_scores = defaultdict(list)
    routes_duration = defaultdict(list)

    # Set a threshold in new model creation flow
    if run_new_model:
        threshold = predict_train_set(lstm,
                                      X_train,
                                      save_model,
                                      add_plots,
                                      threshold,
                                      features_list,
                                      target_features_list,
                                      results_path,
                                      flight_route,
                                      similarity_score,
                                      X_train_scaler,
                                      Y_train,
                                      Y_train_scaler)

    flight_dir = os.path.join(test_data_path, flight_route)
    ATTACKS = get_subdirectories(flight_dir)

    figures_results_path = os.path.join(results_path, "figures")
    create_directories(figures_results_path)

    # Iterate over all attacks in order to find anomalies
    for attack in ATTACKS:
        attacks_path = os.path.join(*[str(test_data_path), str(flight_route), str(attack)])
        for flight_csv in os.listdir(f"{attacks_path}"):

            flight_attack_path = os.path.join(*[str(attacks_path), str(flight_csv)])
            df_test_source = pd.read_csv(f"{flight_attack_path}")
            df_test_labels = df_test_source[[ATTACK_COLUMN]].values

            attack_time = len(df_test_labels)

            input_df_test = df_test_source[features_list]
            target_df_test = df_test_source[target_features_list]

            # Step 1 : Clean test data set
            input_clean_df_test = clean_data(input_df_test)
            target_clean_df_test = clean_data(target_df_test)

            # Step 2: Normalize the data
            X_test = X_train_scaler.transform(input_clean_df_test)

            X_test_preprocessed, Y_test_labels_preprocessed = get_testing_data_lstm(X_test, df_test_labels, window_size)

            Y_test = Y_train_scaler.transform(target_clean_df_test)

            Y_test_preprocessed = get_training_data_lstm(Y_test, window_size)

            X_pred = lstm.predict(X_test_preprocessed, verbose=1)
            assert len(Y_test_preprocessed) == len(X_pred)

            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score_multi(Y_test_preprocessed[i], pred, similarity_score))

            # Add reconstruction error scatter if plots indicator is true
            if add_plots:
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=Y_test_labels_preprocessed,
                                                  threshold=threshold,
                                                  plot_dir=figures_results_path,
                                                  title=f'Outlier Score Testing for {flight_csv} in {flight_route}({attack})')

            predictions = [1 if x >= threshold else 0 for x in scores_test]

            # Add roc curve if plots indicator is true
            if add_plots:
                pass
                # plot_roc(y_true=df_test_labels,y_pred=predictions, plot_dir=results_path,title=f'ROC Curve - {flight_csv} in {flight_route}({attack})')

            attack_start, attack_end = get_attack_boundaries(df_test_source[ATTACK_COLUMN])

            method_scores = get_method_scores(predictions, attack_start, attack_end,
                                              add_window_size=True, window_size=window_size)

            tpr_scores[attack].append(method_scores[0])
            fpr_scores[attack].append(method_scores[1])
            acc_scores[attack].append(method_scores[2])
            delay_scores[attack].append(method_scores[3])
            routes_duration[attack].append(attack_time)

    return tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration


def predict_train_set(lstm,
                      X_train,
                      save_model,
                      add_plots,
                      threshold,
                      features_list,
                      target_features_list,
                      results_path,
                      flight_route,
                      similarity_score,
                      X_train_scaler,
                      Y_train,
                      Y_train_scaler):
    """
    Execute prediction on the train data set
    :param lstm: LSTM model
    :param X_train: train input data frame
    :param save_model: indicator whether the user want to save the model or not
    :param add_plots: indicator whether to add plots or not
    :param threshold: threshold from the train
    :param features_list: the list of features which the user chose for the input
    :param target_features_list: the list of features which the user chose for the target
    :param results_path: the path of results directory
    :param flight_route: current flight route we are working on
    :param similarity_score: similarity function
    :param X_train_scaler: train input normalization scalar
    :param Y_train: train target data frame
    :param Y_train_scaler: train target normalization scalar
    :return: threshold
    """

    X_pred = lstm.predict(X_train, verbose=1)

    scores_train = []

    for i, pred in enumerate(X_pred):
        scores_train.append(anomaly_score_multi(Y_train[i], pred, similarity_score))

    # choose threshold for which <LSTM_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
    threshold = get_threshold(scores_train, threshold)

    # Save created model if the indicator is true
    if save_model:
        data = {}
        data['features'] = features_list
        data['target_features'] = target_features_list
        data['threshold'] = threshold

        model_results_path = os.path.join(results_path, "model_data")
        create_directories(model_results_path)

        model_data_path = os.path.join(*[str(model_results_path), 'model_data.json'])
        with open(f"{model_data_path}", 'w') as outfile:
            json.dump(data, outfile)

        lstm_model_path = os.path.join(*[str(model_results_path), str(flight_route) + '.h5'])
        lstm.save(f"{lstm_model_path}")

        save_input_scaler_file_path = os.path.join(model_results_path, flight_route + "_train_scaler.pkl")
        with open(save_input_scaler_file_path, 'wb') as file:
            pickle.dump(X_train_scaler, file)

        save_target_scaler_file_path = os.path.join(model_results_path, flight_route + "_target_scaler.pkl")
        with open(save_target_scaler_file_path, 'wb') as file:
            pickle.dump(Y_train_scaler, file)

    return threshold
