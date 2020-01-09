import os

import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from utils.shared.lstm_hyper_parameters import lstm_hyper_parameters

import shutil
from datetime import datetime


def cosine_similarity(x, y):
    """
    calculate cosine similarity between 2 given vectors
    :param x: vector
    :param y: vector
    :return: cosine similarity
    """
    return dot(x, y) / (norm(x) * norm(y))


def euclidean_distance(x, y):
    """
    calculate the euclidean distance between 2 given vectors
    :param x: vector
    :param y: vector
    :return: euclidean distance
    """
    return np.linalg.norm(x - y)


def anomaly_score(input_vector, output_vector, similarity_score='Cosine similarity'):
    """
    calculate the anomaly of single output decoder
    :param x: input vector
    :param y: output vector
    :return: anomaly score based on cosine similarity
    """
    assert len(input_vector) == len(output_vector)
    assert len(input_vector) > 0
    assert similarity_score == 'Cosine similarity' or similarity_score == 'euclidean'

    switcher = {
        "Cosine similarity": 1 - cosine_similarity(input_vector, output_vector),
        "euclidean_distance": euclidean_distance(input_vector, output_vector)
    }

    return switcher.get(similarity_score, euclidean_distance(input_vector, output_vector))


def anomaly_score_multi(input_vectors, output_vectors, similarity_score):
    """
    calculate the anomaly of a multiple output decoder
    :param input_vectors: list of input vectors
    :param output_vectors: list of output vectors
    :param similarity_score: name of similarity score function
    :return: anomaly score based on cosine similarity
    """
    sum = 0
    input_length = len(input_vectors)

    assert input_length == len(output_vectors)
    assert input_length > 0

    for i in range(input_length):
        sum += anomaly_score(input_vectors[i], output_vectors[i], similarity_score)

    return sum / input_length


def rolled(list, window_size):
    """
    generator to yield batches of rows from a data frame of <window_size>
    :param list: list
    :param window_size: window size
    :return: batch of rows
    """
    count = 0
    while count <= len(list) - window_size:
        yield list[count: count + window_size]
        count += 1


def get_training_data_lstm(list, window_size):
    """
    get training data for lstm autoencoder
    :param list: the list for training
    :param window_size: window size for each instance in training
    :return: X for training
    """

    X = []
    for val in rolled(list, window_size):
        X.append(val)

    return np.array(X)


def get_testing_data_lstm(list, labels, window_size):
    """
    get testing data for lstm autoencoder
    :param list: the list for testing
    :param labels: labels
    :param window_size: window size for each instance in training
    :return: (X, Y) for testing
    """
    X = []
    for val in rolled(list, window_size):
        X.append(val)

    Y = []
    for val in rolled(labels, window_size):
        Y.append(max(val))

    return np.array(X), np.array(Y)


def get_threshold(scores, percent):
    """
    get threshold for classification from this percent of training set that had lower scores
    (e.g get the threshold error in which 95% of training set had lower values than)
    :param scores:
    :param percent:
    :return: threshold
    """
    assert percent <= 1 and percent > 0

    index = int(len(scores) * percent)

    return sorted(scores)[index - 1]


def get_thresholds(list_scores, percent):
    """
    get threshold for classification from this percent of training set that had lower scores
    (e.g get the threshold error in which 95% of training set had lower values than)
    :param scores: list of scores
    :param percent:
    :return: list of thresholds
    """
    return [get_threshold(scores, percent) for scores in list_scores]


def get_method_scores(prediction, windows):
    """
    get previous method scores (TPR, FPR, delay)
    :param prediction: predictions
    :param windows: list of dictionaries that define lower and upper bounds for attack injections
    :return: TPR, FPR, delay
    """
    fp = 0
    fn = 0
    tp = 0
    tn = 0

    detection_delay = -1

    # for window in windows:
    # upper = window["upper"]
    # lower = window["lower"]

    lower = 180 - lstm_hyper_parameters.get_window_size() + 1
    upper = 249
    assert len(prediction) >= upper
    assert upper > lower

    was_detected = False

    for i in range(lower):
        if prediction[i] == 1:
            fp += 1
        else:
            tn += 1

    for i in range(lower, upper):
        if prediction[i] == 1:
            tp += 1
            if not was_detected:
                was_detected = True
                detection_delay = i - lower
        else:
            fn += 1

    for i in range(upper, len(prediction)):
        if prediction[i] == 1:
            fp += 1
        else:
            tn += 1

    if not was_detected:
        detection_delay = upper - lower

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return tpr, fpr, detection_delay


# def report_results(results_dir_path, verbose=1):
#     """
#
#     :param results_dir_path:
#     :param verbose:
#     :return:
#     """
#     ATTACKS = load_attacks()
#     FLIGHT_ROUTES = load_flight_routes()
#
#     for result in ["nab", "fpr", "tpr", "delay"]:
#         results = pd.DataFrame(columns=ATTACKS)
#         for i, flight_route in enumerate(FLIGHT_ROUTES):
#             df = pd.read_csv(f'{results_dir_path}/{flight_route}_{result}.csv')
#             mean = df.mean(axis=0).values
#             std = df.std(axis=0).values
#             output = [f'{round(x, 2)}±{round(y, 2)}%' for x, y in zip(mean, std)]
#             results.loc[i] = output
#
#         results.index = FLIGHT_ROUTES
#
#         if verbose:
#             print(results)
#
#         results.to_csv(f'{results_dir_path}/final_{result}.csv')

def get_subdirectories(path):
    directories = []
    for directory in os.listdir(path):
        if os.path.isdir(os.path.join(path, directory)):
            directories.append(directory)
    return directories


def create_directories(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.makedirs(path)


def get_current_time():
    now = datetime.now()
    return now.strftime("%b-%d-%Y-%H-%M-%S")


def report_results(results_dir_path, test_data_path, FLIGHT_ROUTES, verbose=1):
    """

    :param results_dir_path:
    :param verbose:
    :return:
    """
    for flight_route in FLIGHT_ROUTES:
        fligth_dir = os.path.join(test_data_path, flight_route)
        ATTACKS = get_subdirectories(fligth_dir)

    for result in ["fpr", "tpr", "delay"]:
        results = pd.DataFrame(columns=ATTACKS)
        for i, flight_route in enumerate(FLIGHT_ROUTES):
            df = pd.read_csv(f'{results_dir_path}/{flight_route}/{flight_route}_{result}.csv')
            mean = df.mean(axis=0).values
            std = df.std(axis=0).values
            output = [f'{round(x, 2)}±{round(y, 2)}%' for x, y in zip(mean, std)]
            results.loc[i] = output

        results.index = FLIGHT_ROUTES

        if verbose:
            print(results)

        results.to_csv(f'{results_dir_path}/final_{result}.csv')


def is_excluded_flight(route, csv):
    """
    return if excluded flight
    :param route: flight route
    :param csv: csv of a flight
    :return:  if excluded
    """
    EXCLUDE_FLIGHTS = load_exclude_flights()

    return route in EXCLUDE_FLIGHTS and csv in EXCLUDE_FLIGHTS[route]


def load_from_yaml(filename, key):
    with open(r'.\\' + filename + '.yaml') as file:
        loaded_file = yaml.load(file, Loader=yaml.FullLoader)
        return loaded_file.get(key)


def load_exclude_flights():
    return load_from_yaml('lstm_model_settings', 'EXCLUDE_FLIGHTS')


def load_attacks():
    return load_from_yaml('names', 'ATTACKS')


def load_flight_routes():
    return load_from_yaml('names', 'FLIGHT_ROUTES')


def plot(data, xlabel, ylabel, title,plot_dir):
    """
    plot
    :param data: the data
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: title
    :return:
    """
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'{plot_dir}/{title}.png')

    #plt.show()

def plot_reconstruction_error_scatter(scores, labels, threshold,plot_dir, title="Outlier Score - After Training",):
    """
    plot reconstruction error as a scatter plot
    :param scores:
    :param labels:
    :param threshold:
    :return:
    """
    plt.figure(figsize=(28, 7))
    plt.scatter(range(len(scores)), scores, c=['k' if label == 1 else 'w' for label in labels],
                edgecolors=['k' if label == 1 else 'y' for label in labels], s=15, alpha=0.4)
    plt.xlabel("Index")
    plt.ylabel("Score")
    plt.title(title)

    plt.hlines(y=threshold, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors='r')


    plt.savefig(f'{plot_dir}/{title}.png')
    #plt.show()
