#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Existing algorithms window which is part of GUI application
'''

import os
import win32api

from tkinter import END
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS, set_path, clear_text
from gui.shared.inputs_validation_helper import is_valid_model_paths, is_valid_model_data_file
from gui.widgets_configurations.helper_methods import set_widget_to_left, set_logo_configuration, \
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


class ExistingAlgorithmsWindow(tk.Frame):
    """
    A Class used to enable the user load existing machine learning model

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle back button click

    set_input_entry()
            Description | Set input entry and update the state

    set_algorithm_path()
            Description | Set the algorithm path in the UI

    next_window()
            Description |  Handle next button click

    validate_next_step()
            Description | Validation before approving the move to the next window

    update_selected_algorithms()
            Description | Updates local variables which algorithms were selected by the user

    set_load_model_parameters()
            Description | Updates input settings and move to next window

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
            text='''Please insert paths for existing models.''')
        set_widget_to_left(self.instructions)

        # Page body
        self.algorithms = dict()
        self.browse_buttons = dict()
        self.input_entries = dict()

        # LSTM existing algorithm
        self.lstm_var = tk.IntVar()
        self.lstm_check_button = tk.Checkbutton(self)
        self.lstm_check_button.place(relx=0.015, rely=0.38, height=32, width=146)
        self.lstm_check_button.configure(text="LSTM",
                                         variable=self.lstm_var,
                                         command=lambda: self.set_input_entry("LSTM", self.lstm_var.get()))
        set_widget_to_left(self.lstm_check_button)

        self.lstm_input = tk.Entry(self)
        self.lstm_input.place(relx=0.195, rely=0.38, height=25, relwidth=0.624)
        self.lstm_input.configure(state='disabled')

        self.lstm_btn = tk.Button(self, command=lambda: self.set_algorithm_path("LSTM"))
        self.lstm_btn.place(relx=0.833, rely=0.38, height=25, width=60)
        self.lstm_btn.configure(state='disabled')
        set_button_configuration(self.lstm_btn, text='''Browse''')

        self.browse_buttons["LSTM"] = self.lstm_btn
        self.input_entries["LSTM"] = self.lstm_input

        # SVR existing algorithm
        self.svr_var = tk.IntVar()
        self.svr_check_button = tk.Checkbutton(self)
        self.svr_check_button.place(relx=0.015, rely=0.47, height=32, width=146)
        self.svr_check_button.configure(text="SVR",
                                        variable=self.svr_var,
                                        command=lambda: self.set_input_entry("SVR", self.svr_var.get()))
        set_widget_to_left(self.svr_check_button)

        self.svr_input = tk.Entry(self)
        self.svr_input.place(relx=0.195, rely=0.47, height=25, relwidth=0.624)
        self.svr_input.configure(state='disabled')

        self.svr_btn = tk.Button(self, command=lambda: self.set_algorithm_path("SVR"))
        self.svr_btn.place(relx=0.833, rely=0.47, height=25, width=60)
        self.svr_btn.configure(state='disabled')
        set_button_configuration(self.svr_btn, text='''Browse''')

        self.browse_buttons["SVR"] = self.svr_btn
        self.input_entries["SVR"] = self.svr_input

        # MLP existing algorithm
        self.mlp_var = tk.IntVar()
        self.mlp_check_button = tk.Checkbutton(self)
        self.mlp_check_button.place(relx=0.015, rely=0.56, height=32, width=146)
        self.mlp_check_button.configure(text="MLP",
                                        variable=self.mlp_var,
                                        command=lambda: self.set_input_entry("MLP",
                                                                             self.mlp_var.get()))
        set_widget_to_left(self.mlp_check_button)

        self.mlp_input = tk.Entry(self)
        self.mlp_input.place(relx=0.195, rely=0.56, height=25, relwidth=0.624)
        self.mlp_input.configure(state='disabled')

        self.mlp_btn = tk.Button(self, command=lambda: self.set_algorithm_path("MLP"))
        self.mlp_btn.place(relx=0.833, rely=0.56, height=25, width=60)
        self.mlp_btn.configure(state='disabled')
        set_button_configuration(self.mlp_btn, text='''Browse''')

        self.browse_buttons["MLP"] = self.mlp_btn
        self.input_entries["MLP"] = self.mlp_input

        # Random Forest existing algorithm
        self.random_forest_var = tk.IntVar()
        self.random_forest_check_button = tk.Checkbutton(self)
        self.random_forest_check_button.place(relx=0.015, rely=0.65, height=32, width=146)
        self.random_forest_check_button.configure(text="Random Forest",
                                                  variable=self.random_forest_var,
                                                  command=lambda: self.set_input_entry("Random Forest",
                                                                                       self.random_forest_var.get()))
        set_widget_to_left(self.random_forest_check_button)

        self.random_forest_input = tk.Entry(self)
        self.random_forest_input.place(relx=0.195, rely=0.65, height=25, relwidth=0.624)
        self.random_forest_input.configure(state='disabled')

        self.random_forest_btn = tk.Button(self, command=lambda: self.set_algorithm_path("Random Forest"))
        self.random_forest_btn.place(relx=0.833, rely=0.65, height=25, width=60)
        self.random_forest_btn.configure(state='disabled')
        set_button_configuration(self.random_forest_btn, text='''Browse''')

        self.browse_buttons["Random Forest"] = self.random_forest_btn
        self.input_entries["Random Forest"] = self.random_forest_input

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

    def reset_widgets(self):
        """
        Reset check bar values
        :return: empty values in the widgets
        """

        widgets = [
            self.lstm_input,
            self.svr_input,
            self.mlp_input,
            self.random_forest_input
        ]

        variables = [
            self.lstm_var,
            self.svr_var,
            self.mlp_var,
            self.random_forest_var
        ]

        check_buttons = [
            self.lstm_check_button,
            self.svr_check_button,
            self.mlp_check_button,
            self.random_forest_check_button
        ]

        for widget in widgets:
            clear_text(widget)
            widget['state'] = 'disabled'

        for var, check_button in zip(variables, check_buttons):
            var.set(0)
            check_button['variable'] = var

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.set_new_model_running(False)
        self.controller.show_frame("MainWindow")

    def set_input_entry(self, entry_name, state):
        """
        Set input entry and update the state
        :param entry_name: input's algorithm name
        :param state: input's state
        :return: updated input
        """

        if state:
            self.browse_buttons[entry_name]['state'] = 'active'
            self.input_entries[entry_name]['state'] = 'normal'
            self.algorithms[entry_name] = ""
        else:
            self.input_entries[entry_name].delete(0, END)
            self.browse_buttons[entry_name]['state'] = 'disabled'
            self.input_entries[entry_name]['state'] = 'disabled'
            self.algorithms.pop(entry_name, None)

    def set_algorithm_path(self, algorithm):
        """
        Set the algorithm path in the UI
        :param algorithm: input algorithm
        :return: updated state
        """

        self.input_entries[algorithm].delete(0, END)
        path = set_path()
        self.input_entries[algorithm].insert(0, path)

    def next_window(self):
        """
        Handle next button click
        :return: if validations pass move to next window
        """

        self.update_selected_algorithms()
        if self.validate_next_step():
            self.set_load_model_parameters()

    def validate_next_step(self):
        """
        Validation before approving the move to the next window
        :return:
        """

        if not self.algorithms:
            win32api.MessageBox(0, 'Please select algorithm & path for the model before the next step.',
                                'Invalid algorithm', 0x00001000)
            return False

        if not is_valid_model_paths(self.algorithms.values()):
            win32api.MessageBox(0, 'At least one of your algorithms paths invalid or not include the required files!',
                                'Invalid inputs', 0x00001000)
            return False

        if not is_valid_model_data_file(self.algorithms.values()):
            win32api.MessageBox(0,
                                'At least one of the required data is missing in model_data json file!',
                                'Missing data', 0x00001000)
            return False

        return True

    def update_selected_algorithms(self):
        """
        Updates local variables which algorithms were selected by the user
        :return: updated selection
        """

        tmp_algorithms = dict()

        for algorithm in self.algorithms:
            tmp_algorithms[algorithm] = self.input_entries[algorithm].get()

        self.algorithms = tmp_algorithms

    def set_load_model_parameters(self):
        """
        Updates input settings and move to next window
        :return: next window
        """

        self.controller.set_existing_algorithms(self.algorithms)
        self.controller.reinitialize_frame("SimilarityFunctionsWindow")
