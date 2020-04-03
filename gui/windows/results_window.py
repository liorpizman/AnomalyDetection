#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from ipython_genutils.py3compat import xrange
from gui.shared.helper_methods import set_widget_for_param, trim_unnecessary_chars
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets.table.table import Table
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
    set_copyright_configuration, set_button_configuration
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


class ResultsWindow(tk.Frame):

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

        self.metrics_table = Table(self,
                                   columns=["Metric", "Down attack", "Up attack", "Fore attack", "Random attack"],
                                   column_minwidths=[None, None, None])
        self.metrics_table.pack(expand=True, fill=X, padx=10, pady=10)

        self.toggle_results_button = tk.Button(self, command=self.toggle_results)
        self.toggle_results_button.place(relx=0.8, rely=0.7, height=25, width=81)
        set_button_configuration(self.toggle_results_button, text='''Change''')

        # Page footer
        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def toggle_results(self):
        pass

    def back_window(self):
        self.controller.show_frame("MainWindow")

    def transform_list(self, source_list):
        transformed_list = []

        for element in source_list:
            transformed_element = trim_unnecessary_chars(element)
            transformed_list.append(transformed_element)

        return transformed_list

    def update_table_view(self, algorithm, flight_route):
        pass

    def reinitialize(self):
        try:
            chosen_algorithms = list(InputSettings.get_algorithms())
            flight_routes = list(InputSettings.get_flight_routes())

            transformed_chosen_algorithms = self.transform_list(chosen_algorithms)
            transformed_flight_routes = self.transform_list(flight_routes)

            original_algorithm = chosen_algorithms[0]
            original_flight_route = flight_routes[0]

            displayed_algorithm = transformed_chosen_algorithms[0]
            displayed_flight_route = transformed_flight_routes[0]

            current_title = 'Results for algorithm: [' + displayed_algorithm + '] and flight route: [' + \
                            displayed_flight_route + ']'

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            results_data = InputSettings.get_results_metrics_data()

            # Algorithm and Flight route permutation choice
            self.parameters = {}

            set_widget_for_param(frame=self,
                                 text="Algorithm:",
                                 combobox_values=transformed_chosen_algorithms,
                                 param_key="algorithm",
                                 relative_x=0.05,
                                 y_coordinate=0.7)

            set_widget_for_param(frame=self,
                                 text="Flight route:",
                                 combobox_values=transformed_flight_routes,
                                 param_key="flight_route",
                                 relative_x=0.4,
                                 y_coordinate=0.7)

            data = results_data[original_algorithm][original_flight_route]  # should be changed to dynamic table

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
