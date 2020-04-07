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
        if file.endswith('.h5') or file.endswith('.pkl'):
            return os.path.join(path, file)
    return ""


def set_widget_for_param(frame, text, combobox_values, param_key, relative_x, y_coordinate):
    try:
        frame.algorithm_param = tk.Label(frame)
        frame.algorithm_param.place(relx=relative_x, rely=y_coordinate, height=25, width=100)
        frame.algorithm_param.configure(text=text)
        set_widget_to_left(frame.algorithm_param)

        frame.algorithm_param_combo = ttk.Combobox(frame, state="readonly", values=combobox_values)
        frame.algorithm_param_combo.place(relx=relative_x + 0.1, rely=y_coordinate, height=25, width=170)
        frame.algorithm_param_combo.current(0)
        frame.parameters[param_key] = frame.algorithm_param_combo
    except Exception as e:
        print("Source: gui/shared/helper_methods.py")
        print("Function: set_widget_for_param")
        print("error: " + str(e))


def trim_unnecessary_chars(text):
    removed_apostrophe = text.replace("'", "")
    removed_underscore = removed_apostrophe.replace("_", " ")
    return removed_underscore.capitalize()


def transform_list(source_list):
    transformed_list = []

    for element in source_list:
        transformed_element = trim_unnecessary_chars(element)
        transformed_list.append(transformed_element)

    return transformed_list


def clear_text(widget):
    widget.delete(0, 'end')
