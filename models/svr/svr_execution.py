import pickle

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from models.svr.svr_hyper_parameters import svr_hyper_parameters
from utils.constants import ATTACK_COLUMN
from utils.routes import *
from utils.helper_methods import get_threshold, report_results, get_method_scores, get_subdirectories, \
    create_directories, get_current_time, \
    plot_reconstruction_error_scatter, get_attack_boundaries, anomaly_score
from sklearn.preprocessing import MaxAbsScaler
from collections import defaultdict
import json


def get_svr_new_model_parameters():
    """
    get svr hyper parameters
    :return:
    """
    return (svr_hyper_parameters.get_kernel(),
            svr_hyper_parameters.get_gamma(),
            svr_hyper_parameters.get_epsilon(),
            svr_hyper_parameters.get_threshold(),)


def get_svr_model(kernel, gamma, epsilon):
    """
    get svr model
    :return: knn model
    """
    return MultiOutputRegressor(SVR(kernel=kernel,
                                    gamma=gamma,
                                    epsilon=epsilon))


def run_model(training_data_path, test_data_path, results_path, similarity_score, save_model, new_model_running,
              algorithm_path, threshold, features_list):
    if new_model_running:
        kernel, gamma, epsilon, threshold = get_svr_new_model_parameters()
    else:
        # To Do - Load model process
        # svr_model = load_model(algorithm_path)
        scalar, X_train = None, None

    FLIGHT_ROUTES = get_subdirectories(test_data_path)

    current_time = get_current_time()

    create_directories(f'{results_path}/svr/{current_time}')

    for similarity in similarity_score:
        create_directories(f'{results_path}/svr/{current_time}/{similarity}')

    for flight_route in FLIGHT_ROUTES:

        if new_model_running:
            svr_model, scalar, X_train = execute_train(flight_route,
                                                       training_data_path=training_data_path,
                                                       results_path=f'{results_path}/svr/{current_time}',
                                                       kernel=kernel,
                                                       gamma=gamma,
                                                       epsilon=epsilon,
                                                       save_model=save_model,
                                                       add_plots=True,
                                                       features_list=features_list)

        for similarity in similarity_score:
            tpr_scores, fpr_scores, delay_scores = execute_predict(flight_route,
                                                                   test_data_path=test_data_path,
                                                                   similarity_score=similarity,
                                                                   threshold=threshold,
                                                                   svr_model=svr_model,
                                                                   scalar=scalar,
                                                                   results_path=f'{results_path}/svr/{current_time}',
                                                                   add_plots=True,
                                                                   run_new_model=new_model_running,
                                                                   X_train=X_train,
                                                                   features_list=features_list)

            current_results_path = f'{results_path}/svr/{current_time}/{similarity}/{flight_route}'
            create_directories(current_results_path)

            df = pd.DataFrame(tpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_tpr.csv', index=False)

            df = pd.DataFrame(fpr_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_fpr.csv', index=False)

            df = pd.DataFrame(delay_scores)
            df.to_csv(f'{current_results_path}/{flight_route}_delay.csv', index=False)

    algorithm_name = "SVR"

    for similarity in similarity_score:
        report_results(f'{results_path}/svr/{current_time}/{similarity}',
                       test_data_path,
                       FLIGHT_ROUTES,
                       algorithm_name)


def execute_train(flight_route,
                  training_data_path=None,
                  results_path=None,
                  kernel=None,
                  gamma=None,
                  epsilon=None,
                  save_model=False,
                  add_plots=True,
                  features_list=None):
    df_train = pd.read_csv(f'{training_data_path}/{flight_route}/without_anom.csv')

    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    svr_model = get_svr_model(kernel=kernel, gamma=gamma, epsilon=epsilon)
    svr_model.fit(X_train, X_train)

    if save_model:
        data = {}
        data['features'] = features_list
        with open(f'{results_path}/model_data.json', 'w') as outfile:
            json.dump(data, outfile)
        save_file_path = os.path.join(results_path, flight_route + ".pkl")
        with open(save_file_path, 'wb') as file:
            pickle.dump(svr_model, file)

    return svr_model, scalar, X_train


def execute_predict(flight_route,
                    test_data_path=None,
                    similarity_score=None,
                    threshold=None,
                    svr_model=None,
                    scalar=None,
                    results_path=None,
                    add_plots=True,
                    run_new_model=False,
                    X_train=None,
                    features_list=None):
    tpr_scores = defaultdict(list)
    fpr_scores = defaultdict(list)
    delay_scores = defaultdict(list)

    if run_new_model:
        X_pred = svr_model.predict(X_train)
        scores_train = []
        for i, pred in enumerate(X_pred):
            scores_train.append(anomaly_score(X_train[i], pred, similarity_score))

        # choose threshold for which <MODEL_THRESHOLD_FROM_TRAINING_PERCENT> % of training were lower
        threshold = get_threshold(scores_train, threshold)

        if add_plots:
            plot_reconstruction_error_scatter(scores=scores_train, labels=[0] * len(scores_train), threshold=threshold,
                                              plot_dir=results_path,
                                              title=f'Outlier Score Training for {flight_route})')

    flight_dir = os.path.join(test_data_path, flight_route)
    ATTACKS = get_subdirectories(flight_dir)

    for attack in ATTACKS:
        for flight_csv in os.listdir(f'{test_data_path}/{flight_route}/{attack}'):

            df_test_source = pd.read_csv(f'{test_data_path}/{flight_route}/{attack}/{flight_csv}')
            Y_test = df_test_source[[ATTACK_COLUMN]].values
            df_test = df_test_source[features_list]

            if not run_new_model:
                scalar = MaxAbsScaler()
                scalar.fit(df_test)

            X_test = scalar.transform(df_test)

            X_pred = svr_model.predict(X_test)

            scores_test = []
            for i, pred in enumerate(X_pred):
                scores_test.append(anomaly_score(X_test[i], pred, similarity_score))

            if add_plots:
                plot_reconstruction_error_scatter(scores=scores_test,
                                                  labels=Y_test,
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
