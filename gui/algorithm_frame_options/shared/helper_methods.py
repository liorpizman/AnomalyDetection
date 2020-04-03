#! /usr/bin/env python
#  -*- coding: utf-8 -*-
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
    relative_x = 0
    try:
        frame.algorithm_param = tk.Label(frame)
        frame.algorithm_param.place(relx=relative_x, rely=y_coordinate, height=25, width=150)
        frame.algorithm_param.configure(text=text)
        set_widget_to_left(frame.algorithm_param)

        frame.algorithm_param_combo = Combobox(frame, state="readonly", values=combobox_values)
        frame.algorithm_param_combo.place(relx=relative_x + 0.35, rely=y_coordinate, height=25, width=150)
        frame.algorithm_param_combo.current(0)
        frame.parameters[param_key] = frame.algorithm_param_combo
    except Exception as e:
        print("Source: gui/algorithm_frame_options/shared/helper_methods.py")
        print("Function: set_widget_for_param")
        print("error: " + str(e))


def load_algorithm_constants(filename):
    path = os.path.join(dirname(abspath(__file__)), filename)
    with open(path) as file:
        algoritms_params = yaml.load(file, Loader=yaml.FullLoader)
        return convert_string_to_boolean(algoritms_params)


def convert_string_to_boolean(source_dict):
    """
    This function converts string values of dictionary to booleans,
    where boolean is present in string
    :param source_dict:
    :return:
    """
    changes = {"True": True,
               "False": False
               }
    for key in source_dict.keys():
        source_dict[key] = [changes.get(x, x) for x in source_dict[key]]
    return source_dict
