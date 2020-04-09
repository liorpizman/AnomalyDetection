#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Methods to handle repeatable actions which are done by algorithms frame
'''

import os
import yaml

from os.path import dirname, abspath
from tkinter.ttk import Combobox
from gui.widgets_configurations.helper_methods import set_widget_to_left

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def set_widget_for_param(frame, text, combobox_values, param_key, y_coordinate):
    """
    Sets a dynamic pair of label and combo box by given parameters
    :param frame: frame to work on
    :param text: label text
    :param combobox_values: possible values for the combo box
    :param param_key:  the key for the pair which will be used in the frame
    :param y_coordinate: y-axis coordinate
    :return: dynamic pair of label and combo box
    """

    relative_x = 0

    try:

        # Create new label
        frame.algorithm_param = tk.Label(frame)
        frame.algorithm_param.place(relx=relative_x, rely=y_coordinate, height=25, width=150)
        frame.algorithm_param.configure(text=text)

        # Set the widget in the left side of the block
        set_widget_to_left(frame.algorithm_param)

        # Create new combo box - possible values for the label
        frame.algorithm_param_combo = Combobox(frame, state="readonly", values=combobox_values)
        frame.algorithm_param_combo.place(relx=relative_x + 0.35, rely=y_coordinate, height=25, width=150)
        frame.algorithm_param_combo.current(0)
        frame.parameters[param_key] = frame.algorithm_param_combo

    except Exception as e:

        # Handle an error with a stack trace print
        print("Source: gui/algorithm_frame_options/shared/helper_methods.py")
        print("Function: set_widget_for_param")
        print("error: " + str(e))


def load_algorithm_constants(filename):
    """
    Loads all the constants for a specific algorithm - according to the filename (which must be the algorithm name)
    :param filename: algorithm name - required
    :return:  all the constant for a given input
    """

    path = os.path.join(dirname(abspath(__file__)), filename)

    with open(path) as file:
        algorithm_params = yaml.load(file, Loader=yaml.FullLoader)
        return convert_string_to_boolean(algorithm_params)


def convert_string_to_boolean(source_dict):
    """
    Converts string values of dictionary to booleans in which boolean values are presented as a string
    :param source_dict: input value with non boolean values
    :return: transformed dictionary with boolean values
    """

    changes = {
        "True": True,
        "False": False
    }

    for key in source_dict.keys():
        source_dict[key] = [changes.get(x, x) for x in source_dict[key]]

    return source_dict
