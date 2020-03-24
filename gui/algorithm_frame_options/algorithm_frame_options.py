#! /usr/bin/env python
#  -*- coding: utf-8 -*-

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
    def __init__(self, parent=None, yaml_filename=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parameters = {}

        self.parameters_lists = load_algorithm_constants(yaml_filename)
        self.parameters_lists_keys = list(self.parameters_lists.keys())

        self.values_lists = []
        for key in self.parameters_lists_keys:
            self.values_lists.append(self.parameters_lists.get(key))

        self.params_texts = self.values_lists.pop(0)
        self.params_keys = self.values_lists.pop(0)

        y_val = 0
        y_delta = 0.14

        index = 0
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
        chosen_params = {}
        for parameter in self.parameters.keys():
            chosen_params[parameter] = self.parameters[parameter].get()
        return chosen_params
