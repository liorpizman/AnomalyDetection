#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
New model window which is part of GUI application
'''

import os

from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import set_path, CROSS_WINDOWS_SETTINGS, clear_text
from gui.widgets_configurations.helper_methods import set_copyright_configuration, set_logo_configuration, \
    set_button_configuration, set_widget_to_left
from gui.shared.inputs_validation_helper import new_model_paths_validation
from tkinter import END

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


class NewModel(tk.Frame):
    """
    A Class used to get all the paths from the user in order to create new models

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    set_input_path()
            Description | Set the train data set path to entry widget

    set_test_path()
            Description | Set the test data set path to entry widget

    set_results_path()
            Description | Set the results path to entry widget

    back_window()
            Description | Handle a click on back button

    next_window()
            Description | Handle a click on next button

    set_new_model_parameters()
            Description | Set parameters to new model flow

    """

    def __init__(self, parent, controller):

        """
        Parameters
        ----------

        :param parent: window
        :param controller: GUI controller
        """

        tk.Frame.__init__(self, parent)

        # Page init
        self.controller = controller
        self.menubar = Menubar(controller)
        # Disables ability to tear menu bar into own window
        self.controller.option_add('*tearOff', 'FALSE')
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
            text='''Please insert 'Mobilicom Ltd' simulated data / ADS-B dataset input files.''')
        set_widget_to_left(self.instructions)

        # Page body

        # Training input directory
        self.training_label = tk.Label(self)
        self.training_label.place(relx=0.015, rely=0.4, height=32, width=146)
        self.training_label.configure(text='''Train directory''')
        set_widget_to_left(self.training_label)

        self.training_input = tk.Entry(self)
        self.training_input.place(relx=0.195, rely=0.4, height=25, relwidth=0.624)

        self.training_btn = tk.Button(self, command=self.set_input_path)
        self.training_btn.place(relx=0.833, rely=0.4, height=25, width=60)
        set_button_configuration(self.training_btn, text='''Browse''')

        # Testing input directory
        self.test_label = tk.Label(self)
        self.test_label.place(relx=0.015, rely=0.5, height=32, width=146)
        self.test_label.configure(text='''Test directory''')
        set_widget_to_left(self.test_label)

        self.test_input = tk.Entry(self)
        self.test_input.place(relx=0.195, rely=0.5, height=25, relwidth=0.624)

        self.test_btn = tk.Button(self, command=self.set_test_path)
        self.test_btn.place(relx=0.833, rely=0.5, height=25, width=60)
        set_button_configuration(self.test_btn, text='''Browse''')

        # Results output directory
        self.results_label = tk.Label(self)
        self.results_label.place(relx=0.015, rely=0.6, height=32, width=146)
        self.results_label.configure(text='''Results directory''')
        set_widget_to_left(self.results_label)

        self.results_input = tk.Entry(self)
        self.results_input.place(relx=0.195, rely=0.6, height=25, relwidth=0.624)

        self.results_btn = tk.Button(self, command=self.set_results_path)
        self.results_btn.place(relx=0.833, rely=0.6, height=25, width=60)
        set_button_configuration(self.results_btn, text='''Browse''')

        # Page footer
        self.next_button = tk.Button(self, command=self.next_window)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Next''')

        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

        # -------------------------------should be replaced at final submission  -------------------------------------
        self.set_inputs_third_permutation()
        # ------------------------------- end ------------------------------------------------------------------------

    def reset_widgets(self):
        """
        Reset check bar values
        :return: empty values in the widgets
        """

        widgets = [
            self.training_input,
            self.test_input,
            self.results_input
        ]

        for widget in widgets:
            clear_text(widget)

    def set_input_path(self):
        """
        Set the train data set path to entry widget
        :return: updated train path
        """

        self.training_input.delete(0, END)
        path = set_path()
        self.training_input.insert(0, path)

    def set_test_path(self):
        """
        Set the test data set path to entry widget
        :return: updated test path
        """

        self.test_input.delete(0, END)
        path = set_path()
        self.test_input.insert(0, path)

    def set_results_path(self):
        """
        Set the results path to entry widget
        :return: updated results path
        """

        self.results_input.delete(0, END)
        path = set_path()
        self.results_input.insert(0, path)

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.set_new_model_running(False)
        self.controller.show_frame("MainWindow")

    def next_window(self):
        """
        Handle a click on next button
        :return: if validations pass move to next window
        """

        if new_model_paths_validation(self.training_input.get(), self.test_input.get(), self.results_input.get()):
            self.set_new_model_parameters()
            self.controller.show_frame("AlgorithmsWindow")

    def set_new_model_parameters(self):
        """
        Set parameters to new model flow
        :return: updated parameters in new model flow in the system
        """

        self.controller.set_new_model_training_input_path(self.training_input.get())
        self.controller.set_new_model_test_input_path(self.test_input.get())
        self.controller.set_new_model_results_input_path(self.results_input.get())
        self.controller.set_features_columns_options()

    # -------------------------------should be replaced at final submission  -------------------------------------

    def set_inputs_first_permutation(self):
        self.set_permutations(
            training_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\train",
            test_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\test",
            results_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\results"
        )

    def set_inputs_second_permutation(self):
        self.set_permutations(
            training_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\ADS-B\\train",
            test_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\ADS-B\\test",
            results_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\ADS-B\\results\\new_model_results"
        )

    def set_inputs_third_permutation(self):
        self.set_permutations(
            training_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\mini_set\\train",
            test_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\mini_set\\test",
            results_path="C:\\Users\\Yehuda Pashay\\Desktop\\flight_data\\data_set\\simulator\\mini_set\\results"
        )

    def set_inputs_fourth_permutation(self):
        self.set_permutations(
            training_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\data_set_simulator\\train",
            test_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\data_set_simulator\\test",
            results_path="C:\\Users\\Lior\\Desktop\\ADS-B Data Set\\data_set_simulator\\results"
        )

    def set_permutations(self, training_path, test_path, results_path):
        self.controller.set_new_model_training_input_path(training_path)
        self.controller.set_new_model_test_input_path(test_path)
        self.controller.set_new_model_results_input_path(results_path)
        self.training_input.insert(0, training_path)
        self.test_input.insert(0, test_path)
        self.results_input.insert(0, results_path)

    # ------------------------------- end ------------------------------------------------------------------------
