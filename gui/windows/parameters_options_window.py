#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os
import win32api

from gui.algorithm_frame_options.algorithm_frame_options import AlgorithmFrameOptions
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import load_anomaly_detection_list
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
    set_button_configuration, set_copyright_configuration

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


class ParametersOptionsWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.menubar = Menubar(controller)
        self.controller.option_add('*tearOff', 'FALSE')  # Disables ability to tear menu bar into own window
        system_logo = CROSS_WINDOWS_SETTINGS.get('LOGO')
        photo_location = os.path.join(system_logo)
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(
            text='''Please select the values for each of the following parameters:''')
        set_widget_to_left(self.instructions)

        # Dynamic algorithm options
        self.algorithms_files = load_anomaly_detection_list()
        self.current_algorithm = self.controller.get_current_algorithm_to_edit()
        self.current_yaml = self.set_suitable_yaml_file(self.current_algorithm)

        self.height_options_frame = 268
        self.width_options_frame = 300

        self.options_to_show = AlgorithmFrameOptions(self, yaml_filename=self.current_yaml)
        self.options_to_show.place(relx=0.1,
                                   rely=0.35,
                                   height=self.height_options_frame,
                                   width=self.width_options_frame)

        # initialize features columns options
        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()
        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.features_listbox = tk.Listbox(self,
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           width=120,
                                           height=200)
        self.features_listbox.place(relx=0.6, rely=0.4, height=200, width=120)

        # Page footer
        self.next_button = tk.Button(self, command=self.handle_next_button)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Save''')

        self.back_button = tk.Button(self, command=lambda: self.controller.show_frame("AlgorithmsWindow"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Cancel''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def handle_next_button(self):
        algorithm_parameters = self.options_to_show.get_algorithm_parameters()
        selected_features = self.get_selected_features()
        self.save_algorithm_parameters(algorithm_parameters, selected_features)

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def set_users_selected_features(self, algorithm_name, features_list):
        self.controller.set_users_selected_features(algorithm_name, features_list)

    def save_algorithm_parameters(self, algorithm_parameters, features_list):
        if not self.validate_next_step():
            return
        else:
            algorithm_name = self.controller.get_current_algorithm_to_edit()
            self.set_algorithm_parameters(algorithm_name, algorithm_parameters)
            self.set_users_selected_features(algorithm_name, features_list)
            self.controller.show_frame("AlgorithmsWindow")

    def validate_next_step(self):
        if not self.get_selected_features():
            win32api.MessageBox(0, 'Please select feature for the algorithm before the next step.', 'Invalid Feature',
                                0x00001000)
            return False
        return True

    def set_suitable_yaml_file(self, algorithm_name):
        switcher = {
            self.algorithms_files[0]: "lstm_params.yaml",
            self.algorithms_files[1]: "ocsvm_params.yaml",
            self.algorithms_files[2]: "knn_params.yaml",
            self.algorithms_files[3]: "isolation_forest_params.yaml",
        }
        return switcher.get(algorithm_name, None)

    def reinitialize(self):
        self.reinitialize_current_algorithm_options()
        self.reinitialize_features_columns_options()

    def reinitialize_features_columns_options(self):
        self.features_label = tk.Label(self)
        self.features_label.place(relx=0.6, rely=0.35, height=32, width=100)
        self.features_label.configure(text='''Select features:''')
        set_widget_to_left(self.features_label)

        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()

        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.features_listbox = tk.Listbox(self,
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           width=120,
                                           height=150)
        self.features_listbox.place(relx=0.6, rely=0.4, height=200, width=120)

    def reinitialize_current_algorithm_options(self):
        self.current_algorithm = self.controller.get_current_algorithm_to_edit()
        self.current_yaml = self.set_suitable_yaml_file(self.current_algorithm)
        self.controller.remove_algorithm(self.current_algorithm)

        self.options_to_show.destroy()
        self.options_to_show = AlgorithmFrameOptions(self, yaml_filename=self.current_yaml)
        self.options_to_show.place(relx=0.1, rely=0.35, height=268, width=450)

    def get_selected_features(self):
        features = []
        selection = self.features_listbox.curselection()
        for i in selection:
            selected = self.features_listbox.get(i)
            features.append(selected)
        return features

    def get_features_columns_options(self):
        return self.controller.get_features_columns_options()
