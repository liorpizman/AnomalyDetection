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
from tkinter.font import Font, ITALIC, BOLD

import yaml

from tkinter import messagebox
from os.path import dirname, abspath
from tkinter.ttk import Combobox
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_widget_to_left, set_info_configuration

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


def set_widget_for_param(frame, text, combobox_values, param_key, y_coordinate, filename):
    """
    Sets a dynamic pair of label and combo box by given parameters
    :param frame: frame to work on
    :param text: label text
    :param combobox_values: possible values for the combo box
    :param param_key:  the key for the pair which will be used in the frame
    :param y_coordinate: y-axis coordinate
    :param filename: the name of the algorithm
    :return: dynamic pair of label and combo box
    """

    relative_x = 0

    try:
        # Creating a photo image object to use for information button
        info_dir = CROSS_WINDOWS_SETTINGS.get('INFORMATION_DIR')
        info_file = CROSS_WINDOWS_SETTINGS.get('INFORMATION_FILE')
        base_folder = os.path.dirname(__file__)
        dir_path = os.path.join(base_folder, info_dir)
        photo_location = os.path.join(dir_path, info_file)
        global info_photo
        info_photo = tk.PhotoImage(file=photo_location)
        # frame.info_png_img = tk.PhotoImage(file=photo_location)

        # Create new label
        frame.algorithm_param = tk.Label(frame)
        frame.algorithm_param.place(relx=relative_x, rely=y_coordinate, height=25, width=150)
        frame.algorithm_param.configure(text=text)

        # Set the widget in the left side of the block
        set_widget_to_left(frame.algorithm_param)

        i_styling = Font(family="Times New Roman",
                         size=12,
                         weight=BOLD,
                         slant=ITALIC)
        frame.algorithm_param_info_button = tk.Button(frame,
                                                      text="i",
                                                      bg="sky blue",
                                                      font=i_styling,
                                                      command=lambda: on_info_button_click(attribute=text,
                                                                                           filename=
                                                                                           filename.replace(
                                                                                               '_params',
                                                                                               ''))
                                                      )
        frame.algorithm_param_info_button.place(relx=relative_x + 0.25, rely=y_coordinate, height=25, width=25)
        # frame.algorithm_param_info_button.configure(image=info_photo)
        # set_info_configuration(frame.algorithm_param_info_button, image=info_photo)

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


def on_info_button_click(attribute, filename):
    """
    Information data for a given attribute
    :param attribute: algorithm parameter
    :param filename: algorithm name
    :return: information data within a message box
    """

    messagebox.askokcancel(
        title='{0} information window'.format(attribute),
        message=load_attribute_data(attribute, filename)
    )


def load_attribute_data(attribute, filename):
    """
    Loads the data about specific attribute for a specific algorithm - according to the filename
    (which must be the algorithm name)
    :param filename: algorithm name - required
    :param attribute: the chosen attribute we want the data about
    :return: detailed data about a given attribute
    """

    sub_path = os.path.join(dirname(abspath(__file__)), 'attributes_information')
    path = os.path.join(sub_path, filename)

    with open(path) as file:
        algorithm_attributes = yaml.load(file, Loader=yaml.FullLoader)
        return algorithm_attributes[attribute]


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
