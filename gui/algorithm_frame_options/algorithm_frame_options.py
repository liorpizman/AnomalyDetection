#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
A frame which represents all the parameters which can be chosen by thw user for each algorithm
'''

from gui.algorithm_frame_options.shared.helper_methods import set_widget_for_param, load_algorithm_constants

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


class AlgorithmFrameOptions(tk.Frame):
    """
    A Class used to map all the actions in the GUI

    Methods
    -------
    get_algorithm_parameters()
            Description | Return all the selected values for each algorithm parameter
                         (selected value from the combo box)

    """

    def __init__(self, parent=None, yaml_filename=None):
        """
        Parameters
        ----------

        parent : ParametersOptionsWindow
            The parent window which controls the current frame

        yaml_filename : str
            The file which suits a specific algorithm in order to construct the frame

        """

        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parameters = {}

        # Load all constant for a given algorithm
        self.parameters_lists = load_algorithm_constants(yaml_filename)
        self.parameters_lists_keys = list(self.parameters_lists.keys())

        # Set values for frame construction
        self.values_lists = []
        for key in self.parameters_lists_keys:
            self.values_lists.append(self.parameters_lists.get(key))

        # Pop keys of each list
        self.params_texts = self.values_lists.pop(0)
        self.params_keys = self.values_lists.pop(0)

        y_val = 0
        y_delta = 0.14

        index = 0

        # Create dynamic pairs of label and combo box for each algorithm param
        for param_description in self.params_texts:
            # Set generic param with suitable values according to suitable yaml file
            set_widget_for_param(frame=self,
                                 text=param_description,
                                 combobox_values=self.values_lists[index],
                                 param_key=self.params_keys[index],
                                 y_coordinate=y_val)
            y_val += y_delta
            index += 1

    def get_algorithm_parameters(self):
        """
        Return all the selected values for each algorithm parameter (selected value from the combo box)
        :return: all parameters' values
        """
        chosen_params = {}
        for parameter in self.parameters.keys():
            chosen_params[parameter] = self.parameters[parameter].get()
        return chosen_params
