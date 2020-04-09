'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Methods to handle repeatable actions which are done by the gui controller
'''

import os
import yaml
import tkinter
import json

from tkinter.filedialog import askdirectory, askopenfilename
from gui.shared.constants import *
from gui.widgets_configurations.helper_methods import set_widget_to_left

try:
    import Tkinter as tk
    from Tkconstants import *
except ImportError:
    import tkinter as tk
    from tkinter.constants import *

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def set_path():
    """
    Set path and return directory name
    :return: directory name
    """

    tkinter.Tk().withdraw()
    directory_name = askdirectory(initialdir=os.getcwd(), title='Please select a directory')

    if len(directory_name) > 0:
        return directory_name
    else:
        return ""


def set_file_path():
    """
    Set path and return file name
    :return: file name
    """

    tkinter.Tk().withdraw()
    file_name = askopenfilename(initialdir=os.getcwd(), title='Please select a file')

    if len(file_name) > 0:
        return file_name
    else:
        return ""


def set_training_path():
    """
    Set the path of the train data set directory
    :return: training path
    """

    global training_path
    training_path = set_path()

    return training_path


def set_test_path():
    """
    Set the path of the test data set directory
    :return: test path
    """

    global test_path
    test_path = set_path()

    return test_path


def load_classification_methods(list_name):
    """
    Load classification methods from a yaml file according to list's key
    :param list_name: key of a list
    :return: list's values
    """

    with open(r'.\shared\classification_methods.yaml') as file:
        classification_methods = yaml.load(file, Loader=yaml.FullLoader)
        return classification_methods.get(list_name)


def load_anomaly_detection_list():
    """
    Load anomaly detection methods
    :return: values of anomaly detection methods
    """

    return load_classification_methods(ANOMALY_DETECTION_METHODS)


def load_similarity_list():
    """
    Load similarity functions
    :return: values of similarity function
    """

    return load_classification_methods(SIMILARITY_FUNCTIONS)


def read_json_file(path):
    """
    Get the data from a JSON file
    :param path:  json file path
    :return: data from a JSON file
    """

    data = None

    with open(path) as json_file:
        data = json.load(json_file)

    return data


def get_model_path(path):
    """
    Get the full path of an existing model (h5/pkl file)
    :param path: model directory path
    :return: full path of the file
    """

    files = os.listdir(path)

    for file in files:
        if file.endswith('.h5') or file.endswith('.pkl'):
            return os.path.join(path, file)

    return ""


def set_widget_for_param(frame, text, combobox_values, param_key, relative_x, y_coordinate):
    """
    Sets a dynamic pair of label and combo box by given parameters
    :param frame: frame to work on
    :param text: label text
    :param combobox_values: possible values for the combo box
    :param param_key:  the key for the pair which will be used in the frame
    :param relative_x: parent relative x
    :param y_coordinate: y-axis coordinate
    :return: dynamic pair of label and combo box
    """

    try:

        # Create new label
        frame.algorithm_param = tk.Label(frame)
        frame.algorithm_param.place(relx=relative_x, rely=y_coordinate, height=25, width=100)
        frame.algorithm_param.configure(text=text)

        # Set the widget in the left side of the block
        set_widget_to_left(frame.algorithm_param)

        # Create new combo box - possible values for the label
        frame.algorithm_param_combo = ttk.Combobox(frame, state="readonly", values=combobox_values)
        frame.algorithm_param_combo.place(relx=relative_x + 0.1, rely=y_coordinate, height=25, width=170)
        frame.algorithm_param_combo.current(0)
        frame.parameters[param_key] = frame.algorithm_param_combo

    except Exception as e:

        # Handle an error with a stack trace print
        print("Source: gui/shared/helper_methods.py")
        print("Function: set_widget_for_param")
        print("error: " + str(e))


def trim_unnecessary_chars(text):
    """
    Remove unnecessary characters in order to present them in the UI
    :param text: text with unnecessary characters
    :return: clean text - text without unnecessary characters
    """

    removed_apostrophe = text.replace("'", "")
    removed_underscore = removed_apostrophe.replace("_", " ")

    return removed_underscore.capitalize()


def transform_list(source_list):
    """
    Clean a list of string in order to present them in the UI
    :param source_list: list of strings with unnecessary characters
    :return: clean list - list of strings without unnecessary characters
    """

    transformed_list = []

    for element in source_list:
        transformed_element = trim_unnecessary_chars(element)
        transformed_list.append(transformed_element)

    return transformed_list


def clear_text(widget):
    """
    Clear the text in an input widget
    :param widget: widget to clean
    :return: clean widget
    """

    widget.delete(0, 'end')
