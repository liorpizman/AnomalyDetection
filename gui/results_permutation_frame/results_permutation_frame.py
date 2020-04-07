#! /usr/bin/env python
#  -*- coding: utf-8 -*-

from ipython_genutils.py3compat import xrange
from gui.widgets.table.table import Table
from gui.shared.helper_methods import set_widget_for_param, trim_unnecessary_chars, transform_list
from gui.widgets_configurations.helper_methods import set_widget_to_left
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


class ResultsPermutationFrame(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.metrics_table = Table(self,
                                   columns=["Metric", "Down attack", "Up attack", "Fore attack", "Random attack"],
                                   column_minwidths=[None, None, None])
        self.metrics_table.pack(expand=True, fill=X, padx=0, pady=0)

        # should be tested - still in the bottom
        # self.metrics_table.place(relx=0, rely=0.1, height=232, width=435)

        try:
            chosen_algorithms = list(InputSettings.get_algorithms())
            flight_routes = list(InputSettings.get_flight_routes())

            transformed_chosen_algorithms = transform_list(chosen_algorithms)
            transformed_flight_routes = transform_list(flight_routes)

            selected_algorithm = self.parent.controller.get_results_selected_algorithm()
            selected_flight_route = self.parent.controller.get_results_selected_flight_route()

            original_algorithm = ""

            for algorithm in chosen_algorithms:
                if trim_unnecessary_chars(algorithm).lower() == selected_algorithm.lower():
                    original_algorithm = algorithm

            original_flight_route = ""

            for route in flight_routes:
                if trim_unnecessary_chars(route).lower() == selected_flight_route.lower():
                    original_flight_route = route

            displayed_algorithm = transformed_chosen_algorithms[0]
            displayed_flight_route = transformed_flight_routes[0]

            current_title = 'Results for algorithm: [{0}] and flight route: [{1}]'.format(displayed_algorithm,
                                                                                          displayed_flight_route)

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0, rely=0, height=32, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            results_data = InputSettings.get_results_metrics_data()

            data = results_data[original_algorithm][original_flight_route]

            attacks_columns = list(data.values())[0]

            # Creates a 2D array, all set to 0
            rows = len(data.keys())
            columns = len(attacks_columns)
            zero_matrix = [[0 for i in xrange(columns)] for i in xrange(rows)]
            self.metrics_table.set_data(zero_matrix)

            for i, metric in enumerate(data.keys()):
                attacks_data = data[metric]
                self.metrics_table.cell(i, 0, metric.upper())
                for j, attack in enumerate(attacks_data.keys()):
                    self.metrics_table.cell(i, j + 1, attacks_data[attack])
        except Exception:
            pass
