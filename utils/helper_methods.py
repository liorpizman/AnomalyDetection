'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Methods to handle repeatable actions which are done by the model controller
'''

import os
import warnings
import math
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from datetime import datetime
from sklearn.metrics import pairwise, mean_squared_error, roc_curve, auc
from utils.constants import NON_ATTACK_VALUE, ATTACK_VALUE


def is_valid_directory(path):
    """
    check whether the input path is a valid path or not
    :param path: string route in the operating system
    :return: true - for a valid path, false - for invalid path
    """

    return os.path.exists(os.path.dirname(path))


def cosine_similarity(x, y):
    """
    calculate cosine similarity between 2 given vectors
    :param x: vector
    :param y: vector
    :return: cosine similarity
    """

    with warnings.catch_warnings():

        warnings.filterwarnings('error')

        try:
            return 1 - (dot(x, y) / (norm(x) * norm(y)))

        except Warning as e:

            x_transformed = x.reshape(-1, 1).transpose((1, 0))
            y_transformed = y.reshape(-1, 1).transpose((1, 0))

            return 1 - pairwise.cosine_similarity(x_transformed, y_transformed)[0][0]


def euclidean_distance(x, y):
    """
    calculate the euclidean distance between 2 given vectors
    :param x: vector
    :param y: vector
    :return: euclidean distance
    """

    return np.linalg.norm(x - y)


def mahalanobis_distance(x, y):
    """
    calculate the mahalanobis distance between 2 given vectors
    :param x: vector
    :param y: vector
    :return: mahalanobis_distance
    """

    v = np.cov(np.array([x, y]).T)
    iv = np.linalg.pinv(v)

    return distance.mahalanobis(x, y, iv)


def mse_distance(x, y):
    """
    calculate the mse distance between 2 given vectors
    :param x: vector
    :param y: vector
    :return: mse distance
    """

    return mean_squared_error(x, y)


def anomaly_score(input_vector, output_vector, similarity_function):
    """
    calculate the anomaly of single output decoder
    :param input_vector: input vector
    :param output_vector: output vector
    :param similarity_function: similarity function method
    :return: anomaly score based on chosen similarity function
    """

    # Input vectors validation
    assert len(input_vector) == len(output_vector)
    assert len(input_vector) > 0
    assert similarity_function == 'Cosine similarity' \
           or similarity_function == 'Euclidean distance' \
           or similarity_function == 'Mahalanobis distance' \
           or similarity_function == 'MSE'

    # Switch between chosen similarity function by the user
    switcher = {
        "Cosine similarity": lambda input_vector, output_vector: cosine_similarity(input_vector, output_vector),
        "Euclidean distance": lambda input_vector, output_vector: euclidean_distance(input_vector, output_vector),
        "Mahalanobis distance": lambda input_vector, output_vector: mahalanobis_distance(input_vector, output_vector),
        "MSE": lambda input_vector, output_vector: mse_distance(input_vector, output_vector)
    }

    return switcher.get(
        similarity_function,
        lambda input_vector, output_vector: cosine_similarity(input_vector, output_vector)
    )(input_vector, output_vector)


def anomaly_score_multi(input_vectors, output_vectors, similarity_function):
    """
    calculate the anomaly of a multiple output prediction
    :param input_vectors: list of input vectors
    :param output_vectors: list of output vectors
    :param similarity_function: name of similarity score function
    :return: anomaly score based on cosine similarity
    """

    sum = 0
    input_length = len(input_vectors)

    # vectors size validations
    assert input_length == len(output_vectors)
    assert input_length > 0

    for i in range(input_length):
        sum += anomaly_score(input_vectors[i], output_vectors[i], similarity_function)

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

    # Percent ranges validation
    assert percent <= 1 and percent > 0

    index = int(len(scores) * percent)

    return sorted(scores)[index - 1]


def get_thresholds(list_scores, percent):
    """
    get threshold for classification from this percent of training set that had lower scores
    (e.g get the threshold error in which 95% of training set had lower values than)
    :param list_scores: list of scores
    :param percent: chosen value by the user
    :return: list of thresholds
    """

    return [get_threshold(scores, percent) for scores in list_scores]


def get_method_scores(prediction, attack_start, attack_end, add_window_size, window_size):
    """
    get previous method scores (tpr, fpr, accuracy, detection delay)
    :param prediction: predictions
    :param run_new_model: Indicator whether the current flow is new model creation or not
    :param attack_start: Index for the first attack raw
    :param attack_end: Index for the last attack raw
    :param add_window_size: Indicator whether to add a window size or not
    :return: tpr, fpr, accuracy, detection delay
    """

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    detection_delay = -1
    lower = attack_start
    upper = attack_end

    # Enrich the process with a window size technique
    if add_window_size:
        assert window_size  # check if window_size != None
        lower = attack_start - window_size + 1
        upper = attack_end - window_size + 1

    # Indexes validation
    assert len(prediction) >= upper
    assert upper > lower

    was_detected = False

    # Proceed the raws without the attack
    for i in range(lower):
        if prediction[i] == 1:
            fp += 1  # Non-anomalous record was classified as a record with anomaly (False Positive)
        else:
            tn += 1  # Non-anomalous record was classified as a record without anomaly (True Negative)

    # Proceed the raws that include the attack
    for i in range(lower, upper):
        if prediction[i] == 1:
            tp += 1  # Anomalous record was classified as a record with anomaly (True Positive)
            if not was_detected:
                was_detected = True
                # Calculate Detection Delay
                detection_delay = i - lower
        else:
            fn += 1  # Anomalous record was classified as a record without anomaly (False Negative)

    # Proceed rest of the raws without the attack
    for i in range(upper, len(prediction)):
        if prediction[i] == 1:
            fp += 1  # Non-anomalous record was classified as a record with anomaly (False Positive)
        else:
            tn += 1  # Non-anomalous record was classified as a record without anomaly (True Negative)

    # In case the attack was not detected
    if not was_detected:
        detection_delay = upper - lower

    # Calculate True Positive Rate (Sensitivity/ Recall / Hit-Rate)
    tpr = tp / (tp + fn)

    # Calculate False Positive Rate (Fall-Out)
    fpr = fp / (fp + tn)

    # Calculate Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)

    return tpr, fpr, acc, detection_delay


def get_attack_boundaries(df_label):
    """
    get the lower and the upper index of the attack according to the classification column in the data set
    :param df_label: the attack column in the data frame
    :return: the lower and upper indexes of the attack
    """

    attack_start = df_label[df_label != NON_ATTACK_VALUE].first_valid_index()
    partial_df = df_label.truncate(before=attack_start)
    if partial_df[partial_df != ATTACK_VALUE].first_valid_index():  # ADS-B Data
        attack_end = partial_df[partial_df != ATTACK_VALUE].first_valid_index() - 1
    else:  # Simulated Data
        attack_end = len(df_label) - 1
    return attack_start, attack_end


def get_subdirectories(path):
    """
    get all sub-directories which are exist in a current path
    :param path: input path
    :return: return all the sub directories in a current path
    """

    directories = []
    for directory in os.listdir(path):
        if os.path.isdir(os.path.join(path, directory)):
            directories.append(directory)

    return directories


def create_directories(path):
    """
    create directories in a given path
    :param path: input os path
    :return: created directories
    """

    if os.path.exists(path) and os.path.isdir(path):
        return
    else:
        os.makedirs(path)


def get_current_time():
    """
    get the current time in the following structure : ["%b-%d-%Y-%H-%M-%S"]
    :return: current time in a string structure
    """

    now = datetime.now()
    return now.strftime("%b-%d-%Y-%H-%M-%S")


def report_results(results_dir_path, test_data_path, FLIGHT_ROUTES, algorithm_name, similarity_function,
                   routes_duration, verbose=1):
    """
    report all the results, according to the algorithm in the input
    :param results_dir_path: the path of results directory
    :param test_data_path: the path of test dataset directory
    :param FLIGHT_ROUTES: names of existing flight routes
    :param algorithm_name: the name of the algorithm that we want to report about
    :param similarity_function: the similarity function we currently report about
    :param routes_duration: routes time duration
    :param verbose: default = 1 , otherwise = can be changed to 0
    :return: all the reports are saved to suitable csv files
    """

    # Set new nested dictionary for a flight route from all the existing flights routes
    from utils.input_settings import InputSettings
    results_data = InputSettings.get_results_metrics_data()

    # Iterate over all existing flight routes in order to present them in the final results table
    for flight_route in FLIGHT_ROUTES:
        flight_dir = os.path.join(test_data_path, flight_route)
        ATTACKS = get_subdirectories(flight_dir)

        try:
            results_data[algorithm_name][flight_route]
        except KeyError:
            results_data[algorithm_name][flight_route] = dict()

        results_data[algorithm_name][flight_route][similarity_function] = dict()

    metrics_list = ['fpr', 'tpr', 'acc', 'delay']

    for metric in metrics_list:

        results = pd.DataFrame(columns=ATTACKS)

        # Iterate over all the flight routes in order to save each results' permutation in a csv file
        for i, flight_route in enumerate(FLIGHT_ROUTES):
            flight_route_metric_path = os.path.join(
                *[str(results_dir_path), str(flight_route), str(flight_route) + '_' + str(metric) + '.csv']
            )
            df = pd.read_csv(f"{flight_route_metric_path}")
            mean = df.mean(axis=0).values
            std = df.std(axis=0).values
            output = []
            for x, y in zip(mean, std):
                if math.isnan(y):
                    output.append(f"{round(x, 2)}")
                else:
                    output.append(f"{round(x, 2)}±{round(y, 2)}%")
            results.loc[i] = output

            results_data[algorithm_name][flight_route][similarity_function][metric] = dict()

            # Iterate over all existing attacks in test data set
            for j, attack in enumerate(ATTACKS):
                results_data[algorithm_name][flight_route][similarity_function][metric][attack] = dict()
                results_data[algorithm_name][flight_route][similarity_function][metric][attack] = output[j]

                results_data[algorithm_name][flight_route][attack + '_duration'] = dict()
                results_data[algorithm_name][flight_route][attack + '_duration'] = \
                    routes_duration[attack][0]

        results.index = FLIGHT_ROUTES

        if verbose:
            print(results)

        # Update evaluated evaluation metric for each attack according to the current algorithm and metric
        InputSettings.update_results_metrics_data(results_data)

        final_metric_path = os.path.join(*[str(results_dir_path), 'final_' + str(metric) + '.csv'])
        results.to_csv(f"{final_metric_path}")


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
    """
    load all the data from a file which is suitable to a given key
    :param filename: the file we want to load from
    :param key: the list we want to load
    :return: list of values according to a given key
    """

    with open(r'.\\' + filename + '.yaml') as file:
        loaded_file = yaml.load(file, Loader=yaml.FullLoader)

        return loaded_file.get(key)


def load_exclude_flights():
    """
    load all the flights which are excluded
    :return: list of the excluded flights
    """

    return load_from_yaml('lstm_model_settings', 'EXCLUDE_FLIGHTS')


def load_attacks():
    """
    load all the attacks in the test set
    :return: list of the attacks
    """

    return load_from_yaml('names', 'ATTACKS')


def load_flight_routes():
    """
    load all the flight routes
    :return: list of the flight routes
    """

    return load_from_yaml('names', 'FLIGHT_ROUTES')


def plot(data, xlabel, ylabel, title, plot_dir):
    """
    plot data by input parameters and save it to a given directory
    :param data: the data we want to present on the plot
    :param xlabel:  x-axis label
    :param ylabel:  y-axis label
    :param title:  plot title
    :param plot_dir: the directory we want to save the plot into
    :return: saved plot in a given directory
    """
    labels = [x for x in range(1, len(data) + 1)]
    plt.plot(labels, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt_path = os.path.join(*[str(plot_dir), str(title) + '.png'])
    plt.savefig(f"{plt_path}")

    plt.clf()

    # plt.show()


def plot_reconstruction_error_scatter(scores, labels, threshold, plot_dir, title="Outlier Score - After Training"):
    """
    plot reconstruction error as a scatter plot and save it to a given directory
    :param scores: input scores
    :param labels: labels for the plot
    :param threshold: which will be shown in the plot
    :param plot_dir: the directory we want to save the plot into
    :param title: the title of the plot
    :return: saved plot in a given directory
    """

    plt.figure(figsize=(28, 7))
    plt.scatter(range(len(scores)), scores, c=['k' if label == 1 else 'w' for label in labels],
                edgecolors=['k' if label == 1 else 'y' for label in labels], s=15, alpha=0.4)
    plt.xlabel("Index")
    plt.ylabel("Score")
    plt.title(title)

    plt.hlines(y=threshold, xmin=plt.xlim()[0], xmax=plt.xlim()[1], colors='r')
    plt_path = os.path.join(*[str(plot_dir), str(title) + '.png'])
    plt.savefig(f"{plt_path}")

    plt.clf()
    # plt.show()


def plot_roc(y_true, y_pred, plot_dir, title="ROC Curve"):
    """
    plot roc curve by computing Area Under the Curve (AUC) using the trapezoidal rule
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot_dir: the directory we want to save the plot into
    :param title: title of the plot
    :return:
    """

    true = y_true.reshape(-1, 1).transpose(1, 0)[0]
    pred = np.array(y_pred)

    fpr, tpr, _ = roc_curve(true, pred)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt_path = os.path.join(*[str(plot_dir), str(title) + '.png'])
    plt.savefig(f"{plt_path}")

    plt.clf()
    # plt.show()


def plot_prediction_performance(Y_train, X_pred, results_path,
                                title, x_label="Index [Second]", y_label="Sensor's Value"):
    """
    plot training performance of model
    :param Y_train: actual data
    :param X_pred: predicted data
    :param results_path: results path
    :param title: plot title
    :param x_label: plot x axis label
    :param y_label: plot y axis label
    :return:
    """

    plt.plot(X_pred, 'darkorange', label="Predicted")
    plt.plot(Y_train, 'navy', label="Actual")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 6)

    results_plt_path = os.path.join(*[str(results_path), str(title) + '.png'])
    plt.savefig(f"{results_plt_path}")

    plt.clf()
    # plt.show()


def multi_mean(X1):
    """
    Calculate mean value of 3D arrays for each sample by the column
    :param X1: data frame
    :return: calculated MSE
    """

    transformed_X1 = np.zeros((X1.shape[0], X1.shape[2]))

    for i, value in enumerate(X1):
        transformed_X1[i] = value.mean(axis=0)

    return transformed_X1
