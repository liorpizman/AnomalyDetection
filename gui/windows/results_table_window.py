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
import pandas as pd

from datetime import datetime
from tkinter import messagebox
from tkinter.font import Font, BOLD
from ipython_genutils.py3compat import xrange
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import trim_unnecessary_chars, transform_list, set_path
from gui.widgets.hover_button import HoverButton
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

    generate_table_columns_list()
            Description | Generates a list of columns for table init by given attacks

    export_table_to_csv(algorithm, similarity_function, flight_route)
            Description | Export current table to csv file

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
        comparison_logo = CROSS_WINDOWS_SETTINGS.get('RESULTS')
        photo_location = os.path.join(system_logo)
        comparison_location = os.path.join(comparison_logo)
        global logo_img, comparison_img
        logo_img = tk.PhotoImage(file=photo_location)
        comparison_img = tk.PhotoImage(file=comparison_location)

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        # Page body
        self.results_table = Table(self,
                                   columns=["Metric", "Down attack", "Up attack", "Fore attack", "Random attack"],
                                   header_anchor=CENTER,
                                   column_minwidths=[1, 1, 1],
                                   pady=2)
        self.results_table.pack(fill=X, padx=18, pady=182)

        self.comparison_png = tk.Button(self)
        self.comparison_png.place(relx=0.7, rely=0.7, height=150, width=145)
        set_logo_configuration(self.comparison_png, image=comparison_img)

        # Page footer
        self.back_button = HoverButton(self, command=self.back_window)
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

            current_title = 'Test set attacks comparison table'

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0.015, rely=0.27, height=35, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            results_data = self.controller.get_results_metrics_data()

            data = results_data[original_algorithm][original_flight_route][original_similarity_function]

            attacks_columns = list(data.values())[0]

            transform_attacks_list = transform_list(list(attacks_columns.keys()))
            table_columns = self.generate_table_columns_list(transform_attacks_list)

            self.results_table.pack_forget()
            # self.results_table.destroy()
            self.results_table = Table(self,
                                       columns=table_columns,
                                       header_anchor=CENTER,
                                       column_minwidths=[1, 1, 1],
                                       pady=2)
            self.results_table.pack(fill=X, padx=18, pady=172)  # pady used to add rows in UI

            # Creates a 2D array, all set to 0
            rows = len(data.keys()) + 2
            columns = len(attacks_columns)
            zero_matrix = [[0 for i in xrange(columns)] for i in xrange(rows)]
            self.results_table.set_data(zero_matrix)

            # Set the updated values to the table
            for i, metric in enumerate(data.keys()):
                attacks_data = data[metric]
                if metric.upper() == 'PARAMS':
                    continue
                if metric.upper() == 'DELAY':
                    self.results_table.cell(i, 0, "Detection delay [sec]")
                else:
                    self.results_table.cell(i, 0, metric.upper())
                for j, attack in enumerate(attacks_data.keys()):
                    self.results_table.cell(i, j + 1, attacks_data[attack])

            self.results_table.cell(rows - 2, 0, "Attack duration [sec]")
            for i in range(1, columns + 1):
                self.results_table.cell(rows - 2, i, str(float(results_data[original_algorithm][original_flight_route][
                                                                   list(attacks_columns.keys())[
                                                                       i - 1] + "_attack_duration"])))

            self.results_table.cell(rows - 1, 0, "Flight duration [sec]")
            for i in range(1, columns + 1):
                self.results_table.cell(rows - 1, i, str(float(results_data[original_algorithm][original_flight_route][
                                                                   list(attacks_columns.keys())[
                                                                       i - 1] + "_duration"])))

            permutation_styling = Font(family="Times New Roman",
                                       size=11,
                                       weight=BOLD)

            self.table_dataframe = pd.DataFrame(self.results_table.get_data(), columns=table_columns)

            self.algorithm_label = tk.Label(self)
            self.algorithm_label.place(relx=0.015, rely=0.69, height=25, width=300)
            self.algorithm_label.configure(
                text="Algorithm: {0}".format(selected_algorithm),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.algorithm_label)

            self.similarity_function_label = tk.Label(self)
            self.similarity_function_label.place(relx=0.015, rely=0.73, height=25, width=300)
            self.similarity_function_label.configure(
                text="Similarity function: {0}".format(selected_similarity_function),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.similarity_function_label)

            self.route_label = tk.Label(self)
            self.route_label.place(relx=0.015, rely=0.77, height=25, width=300)
            self.route_label.configure(
                text="Flight route: {0}".format(selected_flight_route),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.route_label)

            self.export_button = tk.Button(self,
                                           command=lambda: self.export_table_to_csv(
                                               algorithm=selected_algorithm,
                                               similarity_function=selected_similarity_function,
                                               flight_route=selected_flight_route
                                           ))
            self.export_button.place(relx=0.849, rely=0.26, height=25, width=89)
            set_button_configuration(self.export_button, text='''Export table''')
            self.export_button.configure(bg='sandy brown')

            if InputSettings.is_grid_search_dict_empty():
                best_params_state = "disabled"
            else:
                best_params_state = "active"

            self.best_params_button = tk.Button(self,
                                                command=lambda: self.export_best_params_to_csv(
                                                    algorithm=selected_algorithm,
                                                    similarity_function=selected_similarity_function,
                                                    flight_route=selected_flight_route
                                                ))
            self.best_params_button.place(relx=0.35, rely=0.839, height=25, width=220)
            self.best_params_button['state'] = best_params_state
            set_button_configuration(self.best_params_button, text='''Display & Export best algorithm params''')
            self.best_params_button.configure(bg='sandy brown')

        except Exception as e:
            # Handle error in setting new data in the table
            print("Source: gui/windows/results_table_window.py")
            print("Function: reinitialize_results_table")
            print("error: " + str(e))

    def generate_table_columns_list(self, attacks_list):
        """
        Generates a list of columns for table init by given attacks
        :param attacks_list: all existing attacks in the test data set
        :return: full columns list for creating a new results table
        """

        table_columns = ["Metric"]

        for attack in attacks_list:
            table_columns.append(attack)

        return table_columns

    def export_table_to_csv(self, algorithm, similarity_function, flight_route):
        """
        Export current table to csv file
        :return: csv file
        """

        path = set_path()
        current_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
        df_name = "Table_{0}_{1}_{2}_{3}.csv".format(algorithm, similarity_function, flight_route, current_time)
        full_path = os.path.join(*[str(path), df_name])

        try:
            self.table_dataframe.to_csv(full_path, index=False)

            messagebox.askokcancel(
                title="Export to CSV file",
                message="Export to CSV file finished successfully!"
            )
        except:
            messagebox.askokcancel(
                title="Export to CSV file",
                message="Export to CSV file failed! Please try again."
            )

    def export_best_params_to_csv(self, algorithm, similarity_function, flight_route):
        """
        Export best params to csv file
        :return: csv file
        """

        try:
            messagebox.showinfo(
                title="Best Parameters - Grid Search - {0}".format(algorithm.upper()),
                message=str(self.table_dataframe.loc[5])
            )

            path = set_path()
            current_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
            df_name = "Best_Params_{0}_{1}_{2}_{3}.csv".format(algorithm, similarity_function, flight_route,
                                                               current_time)
            full_path = os.path.join(*[str(path), df_name])

            self.table_dataframe.loc[5].to_csv(full_path, index=False)

            messagebox.askokcancel(
                title="Export to CSV file",
                message="Export to CSV file finished successfully!"
            )
        except:
            messagebox.askokcancel(
                title="Export to CSV file",
                message="Export to CSV file failed! Please try again."
            )
