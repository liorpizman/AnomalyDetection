#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
MLP train and prediction execution function
'''

import pickle
import json
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor

from gui.algorithm_frame_options.shared.helper_methods import get_grid_params
from models.data_preprocessing.data_cleaning import clean_data
from models.data_preprocessing.data_normalization import normalize_data
from models.mlp.mlp_hyper_parameters import mlp_hyper_parameters
from models.time_series_model.time_series_estimator import TimeSeriesRegressor
from utils.constants import ATTACK_COLUMN
from utils.input_settings import InputSettings
from utils.routes import *
from utils.helper_methods import get_threshold, report_results, get_method_scores, get_subdirectories, \
    create_directories, get_current_time, \
    plot_reconstruction_error_scatter, get_attack_boundaries, anomaly_score, plot_prediction_performance, get_plots_key, \
    calculate_auc, get_auc_plot_key, get_estimator, tuning_auc
from collections import defaultdict


def get_mlp_new_model_parameters():
    """
    Get MLP hyper parameters
    :return: MLP hyper parameters
    """

    return (
        mlp_hyper_parameters.get_hidden_layer_sizes(),
        mlp_hyper_parameters.get_activation(),
        mlp_hyper_parameters.get_solver(),
        mlp_hyper_parameters.get_alpha(),
        mlp_hyper_parameters.get_random_state(),
        mlp_hyper_parameters.get_threshold(),
        mlp_hyper_parameters.get_window_size()
    )


def get_mlp_parameters_dictionary():
    """
    Get MLP hyper parameters dictionary
    :return: MLP hyper parameters dictionary
    """

    parameters = dict()

    parameters["hidden_layer_sizes"] = mlp_hyper_parameters.get_hidden_layer_sizes()
    parameters["activation"] = mlp_hyper_parameters.get_activation()
    parameters["solver"] = mlp_hyper_parameters.get_solver()
    parameters["alpha"] = mlp_hyper_parameters.get_alpha()
    parameters["random_state"] = mlp_hyper_parameters.get_random_state()
    parameters["threshold percent"] = mlp_hyper_parameters.get_threshold()
    parameters["window size"] = mlp_hyper_parameters.get_window_size()

    return parameters


def get_mlp_model(hidden_layer_sizes, activation, solver, alpha, random_state):
    """
    Get MLP model
    :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    :param activation: Activation function for the hidden layer.
    :param solver: The solver for weight optimization.
    :param alpha: L2 penalty (regularization term) parameter
    :param random_state: If int, random_state is the seed used by the random number generator
    :return: MLP model
    """

    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        random_state=random_state,
                        shuffle=False)


def run_model(training_data_path, test_data_path, results_path, similarity_score, save_model, new_model_running,
              algorithm_path, threshold, features_list, target_features_list, train_scaler_path, target_scaler_path,
              event):
    """
    Run MLP model process
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
    :param event: running state flag
    :return:  reported results for MLP execution
    """

    grid_dictionary = get_grid_params("MLP")
    print('grid_dictionary: {0}'.format(grid_dictionary))
    # Choose between new model creation flow and load existing model flow
    if new_model_running:
        hidden_layer_sizes, activation, solver, alpha, \
        random_state, threshold, window_size = get_mlp_new_model_parameters()
    else:
        mlp_model = pickle.load(open(algorithm_path, 'rb'))
        X_train_scaler = pickle.load(open(train_scaler_path, 'rb'))
        Y_train_scaler = pickle.load(open(target_scaler_path, 'rb'))
        window_size = mlp_model.n_prev
        X_train = None
        Y_train = None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    current_time_path = os.path.join(*[str(results_path), 'MLP', str(current_time)])
    create_directories(f"{current_time_path}")

    # Create sub directories for each similarity function
    for similarity in similarity_score:
        similarity_path = os.path.join(*[str(current_time_path), str(similarity)])
        create_directories(f"{similarity_path}")

    # Train the model for each flight route
    for flight_route in FLIGHT_ROUTES:

        # Execute training for new model flow
        if new_model_running:
            mlp_model, X_train_scaler, Y_train_scaler, X_train, Y_train = execute_train(flight_route,
                                                                                        training_data_path=training_data_path,
                                                                                        hidden_layer_sizes=hidden_layer_sizes,
                                                                                        activation=activation,
                                                                                        solver=solver,
                                                                                        alpha=alpha,
                                                                                        random_state=random_state,
                                                                                        features_list=features_list,
                                                                                        window_size=window_size,
                                                                                        target_features_list=target_features_list,
                                                                                        event=event)

        # Get results for each similarity function
        for similarity in similarity_score:
            current_results_path = os.path.join(*[str(current_time_path), str(similarity), str(flight_route)])
            create_directories(f"{current_results_path}")
            tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration, attack_duration, auc_scores, best_params = execute_predict(
                flight_route,
                test_data_path=test_data_path,
                similarity_score=similarity,
                threshold=threshold,
                mlp_model=mlp_model,
                X_train_scaler=X_train_scaler,
                results_path=current_results_path,
                add_plots=True,
                run_new_model=new_model_running,
                X_train=X_train,
                features_list=features_list,
                target_features_list=target_features_list,
                save_model=save_model,
                Y_train_scaler=Y_train_scaler,
                Y_train=Y_train,
                window_size=window_size,
                event=event,
                grid_dictionary=grid_dictionary
            )

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

            df = pd.DataFrame(auc_scores)
            auc_path = os.path.join(*[str(current_results_path), str(flight_route) + '_auc.csv'])
            df.to_csv(f"{auc_path}", index=False)

            df = pd.DataFrame(best_params)
            best_params_path = os.path.join(*[str(current_results_path), str(flight_route) + '_params.csv'])
            df.to_csv(f"{best_params_path}", index=False)

    algorithm_name = "MLP"

    # Report results for training data to csv files
    for similarity in similarity_score:
        report_similarity_path = os.path.join(*[str(results_path), 'MLP', str(current_time), str(similarity)])
        report_results(f"{report_similarity_path}",
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name,
                       similarity,
                       routes_duration,
                       attack_duration)


def execute_train(flight_route,
                  training_data_path=None,
                  hidden_layer_sizes=None,
                  activation=None,
                  solver=None,
                  alpha=None,
                  random_state=None,
                  features_list=None,
                  window_size=1,
                  target_features_list=None,
                  event=None):
    """
    Execute train function for a specific flight route
    :param flight_route: current flight route we should train on
    :param training_data_path: the path of training data directory
    :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    :param activation: Activation function for the hidden layer.
    :param solver: The solver for weight optimization.
    :param alpha: L2 penalty (regularization term) parameter.
    :param random_state: If int, random_state is the seed used by the random number generator
    :param features_list: the list of features which the user chose for the train
    :param window_size: window size for each instance in training
    :param target_features_list: the list of features which the user chose for the target
    :param event: running state flag
    :return: MLP model, normalization input train scalar,normalization input target scalar, X_train data frame,Y_train data frame
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

    Y_train, Y_train_scaler = normalize_data(data=target_df_train,  # target data
                                             scaler="min_max")

    # Get the model which is created by user's parameters
    mlp_model = get_mlp_model(hidden_layer_sizes=hidden_layer_sizes,
                              activation=activation,
                              solver=solver,
                              alpha=alpha,
                              random_state=random_state)

    tsr = TimeSeriesRegressor(mlp_model, n_prev=window_size)

    event.wait()

    tsr.fit(X_train, Y_train)

    return tsr, X_train_scaler, Y_train_scaler, X_train, Y_train


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    threshold=None,
                    mlp_model=None,
                    X_train_scaler=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None,
                    target_features_list=None,
                    save_model=False,
                    Y_train_scaler=None,
                    Y_train=None,
                    window_size=None,
                    event=None,
                    grid_dictionary=None):
    """
    Execute predictions function for a specific flight route
    :param flight_route: current flight route we should train on
    :param test_data_path: the path of test data directory
    :param similarity_score: similarity function
    :param threshold: threshold from the train
    :param mlp_model: MLP model
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
    :param window_size: window size for each instance in training
    :param event: running state flag
    :param grid_dictionary: grid search parameters
    :return: tpr scores, fpr scores, acc scores, delay scores, routes duration, attack duration, auc_scores
    """

    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    acc_scores = defaultdict(list)
    delay_scores = defaultdict(list)
    routes_duration = defaultdict(list)
    attack_duration = defaultdict(list)
    auc_scores = defaultdict(list)
    best_params = defaultdict(list)

    # Set a threshold in new model creation flow
    if run_new_model:
        event.wait()
        threshold = predict_train_set(mlp_model,
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

    figures_results_path = os.path.join(results_path, "Figures")
    create_directories(figures_results_path)

    attacks_figures_results_path = os.path.join(figures_results_path, "Attacks")
    create_directories(attacks_figures_results_path)

    # Iterate over all attacks in order to find anomalies
    for attack in ATTACKS:
        event.wait()
        attack_name = attack

        if "_" in attack_name:
            attack_name = attack_name.split("_")[0]

        current_attack_figures_results_path = os.path.join(attacks_figures_results_path, attack_name)
        create_directories(current_attack_figures_results_path)

        attacks_path = os.path.join(*[str(test_data_path), str(flight_route), str(attack)])
        for flight_csv in os.listdir(f"{attacks_path}"):

            flight_attack_path = os.path.join(*[str(attacks_path), str(flight_csv)])
            df_test_source = pd.read_csv(f"{flight_attack_path}")

            Y_test_labels = df_test_source[[ATTACK_COLUMN]].values
            Y_test_labels_preprocessed = mlp_model._preprocess(Y_test_labels, Y_test_labels)[1]

            attack_time = len(Y_test_labels)

            input_df_test = df_test_source[features_list]
            target_df_test = df_test_source[target_features_list]

            # Step 1 : Clean test data set
            input_clean_df_test = clean_data(input_df_test)
            target_clean_df_test = clean_data(target_df_test)

            # Step 2: Normalize the data
            X_test = X_train_scaler.transform(input_clean_df_test)

            Y_test = Y_train_scaler.transform(target_clean_df_test)

            Y_test_preprocessed = mlp_model._preprocess(Y_test, Y_test)[1]

            current_best_params = {}
            if grid_dictionary:
                mlp_model, current_best_params = get_gridSearch_model(grid_dictionary, X_test, Y_test_labels,
                                                                      window_size,
                                                                      X_train, Y_train)

            X_pred = mlp_model.predict(X_test)
            assert len(Y_test_preprocessed) == len(X_pred)

            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score(Y_test_preprocessed[i], pred, similarity_score))

            # Add reconstruction error scatter if plots indicator is true
            event.wait()
            if add_plots:
                title = f'Outlier Score Testing for {flight_csv} in {flight_route}({attack})'
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=Y_test_labels_preprocessed,
                                                  threshold=threshold,
                                                  plot_dir=current_attack_figures_results_path,
                                                  title=title
                                                  )
                key = get_plots_key(algorithm='MLP', similarity=similarity_score, flight_route=flight_route)
                plt_path = os.path.join(*[str(current_attack_figures_results_path), str(title) + '.png'])
                InputSettings.set_plots(key, plt_path)

                for i, target_feature in enumerate(target_features_list):
                    title = "Test performance of MLP for " + target_feature + " feature in " + flight_csv
                    plot_prediction_performance(Y_train=Y_test_preprocessed[:, i],
                                                X_pred=X_pred[:, i],
                                                results_path=current_attack_figures_results_path,
                                                title=title)

            predictions = [1 if x >= threshold else 0 for x in scores_test]

            # Add roc curve if plots indicator is true
            if add_plots:
                pass
                # plot_roc(y_true=Y_test,y_pred=predictions, plot_dir=results_path,title=f'ROC Curve - {flight_csv} in {flight_route}({attack})')

            attack_start, attack_end = get_attack_boundaries(df_test_source[ATTACK_COLUMN])

            method_scores = get_method_scores(predictions, attack_start, attack_end,
                                              add_window_size=True, window_size=window_size)

            auc_key = get_auc_plot_key(algorithm='MLP', similarity=similarity_score, flight_route=flight_route)
            auc_title = f'Receiver Operating Characteristic for {flight_csv} in {flight_route}({attack})'
            auc_plt_path = os.path.join(*[str(current_attack_figures_results_path), str(auc_title) + '.png'])
            InputSettings.set_plots(auc_key, auc_plt_path)
            auc = calculate_auc(scores_test, Y_test_labels_preprocessed.ravel(), 'MLP', auc_plt_path, attack)

            tpr_scores[attack].append(method_scores[0])
            fpr_scores[attack].append(method_scores[1])
            acc_scores[attack].append(method_scores[2])
            delay_scores[attack].append(method_scores[3])
            routes_duration[attack].append(attack_time)
            attack_duration[attack].append(method_scores[4])
            auc_scores[attack].append(auc)
            best_params[attack].append(current_best_params)

    return tpr_scores, fpr_scores, acc_scores, delay_scores, routes_duration, attack_duration, auc_scores, best_params


def predict_train_set(mlp_model,
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
    :param mlp_model: MLP model
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

    X_pred = mlp_model.predict(X_train)
    scores_train = []

    Y_train_preprocessed = mlp_model._preprocess(Y_train, Y_train)[1]
    assert len(Y_train_preprocessed) == len(X_pred)

    for i, pred in enumerate(X_pred):
        scores_train.append(anomaly_score(Y_train_preprocessed[i], pred, similarity_score))

    # choose threshold for which <MODEL_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
    threshold = get_threshold(scores_train, threshold)

    figures_results_path = os.path.join(results_path, "Figures")
    create_directories(figures_results_path)

    if add_plots:

        train_figures_results_path = os.path.join(figures_results_path, "Train")
        create_directories(train_figures_results_path)

        for i, target_feature in enumerate(target_features_list):
            title = "Training performance of MLP for " + target_feature + " in " + flight_route
            plot_prediction_performance(Y_train=Y_train_preprocessed[:, i],
                                        X_pred=X_pred[:, i],
                                        results_path=train_figures_results_path,
                                        title=title)

    # Save created model if the indicator is true
    if save_model:
        data = {}
        data['features'] = features_list
        data['target_features'] = target_features_list
        data['threshold'] = threshold
        data['params'] = get_mlp_parameters_dictionary()

        model_results_path = os.path.join(results_path, "model_data")
        create_directories(model_results_path)

        model_data_path = os.path.join(*[str(model_results_path), 'model_data.json'])
        with open(f"{model_data_path}", 'w') as outfile:
            json.dump(data, outfile)

        save_model_file_path = os.path.join(model_results_path, flight_route + "_model.pkl")
        with open(save_model_file_path, 'wb') as file:
            pickle.dump(mlp_model, file)

        save_input_scaler_file_path = os.path.join(model_results_path, flight_route + "_train_scaler.pkl")
        with open(save_input_scaler_file_path, 'wb') as file:
            pickle.dump(X_train_scaler, file)

        save_target_scaler_file_path = os.path.join(model_results_path, flight_route + "_target_scaler.pkl")
        with open(save_target_scaler_file_path, 'wb') as file:
            pickle.dump(Y_train_scaler, file)

    return threshold


def get_best_model(best_params_, X_train, Y_train, window_size):
    """
    get best model according to gridSearch result
    :param best_params_: model params
    :param X_train: X_train
    :param Y_train: Y_train
    :param window_size: window_size
    :return: best model , best params
    """

    tsr = TimeSeriesRegressor(MLPRegressor(**best_params_), n_prev=window_size)
    tsr.fit(X_train, Y_train)
    return tsr, best_params_


def get_gridSearch_model(grid_dictionary, X_test, Y_test_labels, window_size, X_train, Y_train):
    """
    run algorithm gridSearch
    :param grid_dictionary: params dictionary
    :param X_test: X_test
    :param Y_test_labels: Y_test_labels
    :param window_size: window_size
    :param X_train: X_train
    :param Y_train: Y_train
    :return: best model , best params
    """
    estimator = get_estimator("MLP")
    tsr = TimeSeriesRegressor(estimator, n_prev=window_size)
    grid_auc = make_scorer(tuning_auc, greater_is_better=True, needs_threshold=True)

    grid_search_model = GridSearchCV(tsr, param_grid=grid_dictionary, scoring=grid_auc)
    grid_search_model.fit(X_test, Y_test_labels)

    return get_best_model(grid_search_model.best_params_, X_train, Y_train, window_size)
