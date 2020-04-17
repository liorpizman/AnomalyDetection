#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Results table window which is part of GUI application
'''

import os

from ipython_genutils.py3compat import xrange
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import trim_unnecessary_chars
from gui.widgets.menubar import Menubar
from gui.widgets.table.table import Table
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_button_configuration, \
    set_copyright_configuration, set_widget_to_left
from utils.input_settings import InputSettings

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


class ResultsTableWindow(tk.Frame):
    """
    A Class used to present a results table permutation by an algorithm and a flight route

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle a click on back button

    reinitialize()
            Description | Reinitialize frame values and view

    reinitialize_results_table()
            Description | Reinitialize results table and view

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
        self.results_table = Table(self,
                                   columns=["Metric", "Down attack", "Up attack", "Fore attack", "Random attack"],
                                   column_minwidths=[None, None, None])
        self.results_table.pack(expand=True, fill=X, padx=0, pady=0)

        # Page footer
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

        pass

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.reinitialize_frame("ResultsWindow")

    def reinitialize(self):
        """
         Reinitialize frame values and view
         :return: new frame view
         """

        self.reinitialize_results_table()

    def reinitialize_results_table(self):
        """
         Reinitialize results table and view
         :return: new frame view
         """

        try:
            # Handle suitable flow
            new_model_running = self.controller.get_new_model_running()

            if new_model_running:
                chosen_algorithms = list(self.controller.get_algorithms())
            else:
                chosen_algorithms = list(self.controller.get_existing_algorithms().keys())

            flight_routes = list(self.controller.get_flight_routes())
            similarity_functions = list(self.controller.get_similarity_functions())

            # selected values with transformation to UI components
            selected_algorithm = self.controller.get_results_selected_algorithm()
            selected_flight_route = self.controller.get_results_selected_flight_route()
            selected_similarity_function = self.controller.get_results_selected_similarity_function()

            original_algorithm = ""

            for algorithm in chosen_algorithms:
                if trim_unnecessary_chars(algorithm).lower() == selected_algorithm.lower():
                    original_algorithm = algorithm

            original_flight_route = ""

            for route in flight_routes:
                if trim_unnecessary_chars(route).lower() == selected_flight_route.lower():
                    original_flight_route = route

            original_similarity_function = ""

            for similarity_function in similarity_functions:
                if trim_unnecessary_chars(similarity_function).lower() == selected_similarity_function.lower():
                    original_similarity_function = similarity_function

            current_title = 'Results for algorithm: [{0}], flight route: [{1}] and similarity function: [{2}] '.format(
                selected_algorithm,
                selected_flight_route,
                selected_similarity_function
            )

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            results_data = self.controller.get_results_metrics_data()

            data = results_data[original_algorithm][original_flight_route][original_similarity_function]

            attacks_columns = list(data.values())[0]

            # Creates a 2D array, all set to 0
            rows = len(data.keys())
            columns = len(attacks_columns)
            zero_matrix = [[0 for i in xrange(columns)] for i in xrange(rows)]
            self.results_table.set_data(zero_matrix)

            # Set the updated values to the table
            for i, metric in enumerate(data.keys()):
                attacks_data = data[metric]
                self.results_table.cell(i, 0, metric.upper())
                for j, attack in enumerate(attacks_data.keys()):
                    self.results_table.cell(i, j + 1, attacks_data[attack])

        except Exception:
            # Handle error in setting new data in the table
            pass
