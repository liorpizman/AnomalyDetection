import os
import yaml
import tkinter
import json

from tkinter.filedialog import askdirectory, askopenfilename
from gui.shared.constants import *


def set_path():
    tkinter.Tk().withdraw()
    dirname = askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    if len(dirname) > 0:
        return dirname
    else:
        return ""


def set_file_path():
    tkinter.Tk().withdraw()
    dirname = askopenfilename(initialdir=os.getcwd(), title='Please select a file')
    if len(dirname) > 0:
        return dirname
    else:
        return ""


def set_training_path():
    global training_path
    training_path = set_path()
    return training_path


def set_test_path():
    global test_path
    test_path = set_path()
    return test_path


def load_classification_methods(list_name):
    with open(r'.\shared\classification_methods.yaml') as file:
        classification_methods = yaml.load(file, Loader=yaml.FullLoader)
        return classification_methods.get(list_name)


def load_anomaly_detection_list():
    return load_classification_methods(ANOMALY_DETECTION_METHODS)


def load_similarity_list():
    return load_classification_methods(SIMILARITY_FUNCTIONS)


def read_json_file(path):
    data = None
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def get_model_path(path):
    files = os.listdir(path)
    for file in files:
        if file.endswith('.h5'):
            return os.path.join(path, file)
    return ""
