#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Results window which is part of GUI application
'''

import os

from gui.shared.helper_methods import set_widget_for_param, transform_list
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
    set_copyright_configuration, set_button_configuration

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


class ResultsWindow(tk.Frame):
    """
    A Class used to enable the user to choose a permutation of a results table

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle a click on back button

    toggle_results()
            Description | Toggle permutation of results

    reinitialize()
            Description | Reinitialize frame values and view

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

        # Page body
        self.toggle_results_button = tk.Button(self, command=self.toggle_results)
        self.toggle_results_button.place(relx=0.67, rely=0.5, height=25, width=81)
        set_button_configuration(self.toggle_results_button, text='''Show results''')

        # Page footer
        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Home page''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def reset_widgets(self):
        """
        Reset check bar values
        :return: empty values in the widgets
        """

        pass

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.reset_frame()
        self.controller.reset_input_settings_params()
        self.controller.show_frame("MainWindow")

    def toggle_results(self):
        """
        Toggle permutation of results
        :return: updated permutation which was selected by the user
        """
        selected_algorithm = self.parameters['algorithm'].get()
        selected_flight_route = self.parameters['flight_route'].get()
        selected_similarity_function = self.parameters['similarity_function'].get()

        self.controller.set_results_selected_algorithm(selected_algorithm)
        self.controller.set_results_selected_flight_route(selected_flight_route)
        self.controller.set_results_selected_similarity_function(selected_similarity_function)

        self.controller.reinitialize_frame("ResultsTableWindow")

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        new_model_running = self.controller.get_new_model_running()
        if new_model_running:
            chosen_algorithms = list(self.controller.get_algorithms())
        else:
            chosen_algorithms = list(self.controller.get_existing_algorithms().keys())

        flight_routes = list(self.controller.get_flight_routes())
        similarity_functions = list(self.controller.get_similarity_functions())

        transformed_chosen_algorithms = transform_list(chosen_algorithms)
        transformed_flight_routes = transform_list(flight_routes)
        transformed_similarity_functions = transform_list(similarity_functions)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(text="Choose an algorithm and a flight route in order to get the results.")
        set_widget_to_left(self.instructions)

        # Algorithm and Flight route permutation choice
        self.parameters = {}

        # set dynamic pair of label and combo box to select an algorithm
        set_widget_for_param(frame=self,
                             text="Algorithm:",
                             combobox_values=transformed_chosen_algorithms,
                             param_key="algorithm",
                             relative_x=0.05,
                             y_coordinate=0.4)

        # set dynamic pair of label and combo box to select a flight route
        set_widget_for_param(frame=self,
                             text="Flight route:",
                             combobox_values=transformed_flight_routes,
                             param_key="flight_route",
                             relative_x=0.45,
                             y_coordinate=0.4)

        # set dynamic pair of label and combo box to select a similarity function
        set_widget_for_param(frame=self,
                             text="Similarity:",
                             combobox_values=transformed_similarity_functions,
                             param_key="similarity_function",
                             relative_x=0.05,
                             y_coordinate=0.45)
