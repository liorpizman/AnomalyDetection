#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Results plot window which is part of GUI application
'''

import os

from tkinter.font import Font, BOLD
from ipython_genutils.py3compat import xrange
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import trim_unnecessary_chars, transform_list
from gui.widgets.hover_button import HoverButton
from gui.widgets.menubar import Menubar
from gui.widgets.table.table import Table
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_button_configuration, \
    set_copyright_configuration, set_widget_to_left
from utils.helper_methods import get_plots_key
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


class ResultsPlotWindow(tk.Frame):
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

    reinitialize_results_plot()
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

        self.reinitialize_results_plot()

    def reinitialize_results_plot(self):
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

            current_title = 'Test set attacks plots'

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0.015, rely=0, height=35, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            # Page header
            plot_key = get_plots_key(original_algorithm, original_similarity_function, original_flight_route)
            self.plots_list = InputSettings.get_plots(plot_key)
            self.plots_index = 0

            self.plot_buttons = list()
            for image_plot_path in self.plots_list:

            # plot_location = self.plots_list[self.plots_index]
            # global plot_img
            #
            # plot_img = tk.PhotoImage(file=plot_location)

            # new_width = 500
            # old_width = 2688
            # new_height = 180
            # old_height = 672
            # scale_w = new_width / old_width
            # scale_h = new_height / old_height
            # plot_img.zoom(scale_w, scale_h)

            #plot_img = plot_img.subsample(4, 2)

            # self.plot_png = tk.Button(self)
            # self.plot_png.place(relx=0.015, rely=0.05, height=336, width=672)
            # set_logo_configuration(self.plot_png, image=plot_img)

            permutation_styling = Font(family="Times New Roman",
                                       size=11,
                                       weight=BOLD)

            self.algorithm_label = tk.Label(self)
            self.algorithm_label.place(relx=0.35, rely=0.72, height=25, width=300)
            self.algorithm_label.configure(
                text="Algorithm: {0}".format(selected_algorithm),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.algorithm_label)

            self.similarity_function_label = tk.Label(self)
            self.similarity_function_label.place(relx=0.35, rely=0.76, height=25, width=300)
            self.similarity_function_label.configure(
                text="Similarity function: {0}".format(selected_similarity_function),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.similarity_function_label)

            self.route_label = tk.Label(self)
            self.route_label.place(relx=0.35, rely=0.8, height=25, width=300)
            self.route_label.configure(
                text="Flight route: {0}".format(selected_flight_route),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.route_label)


        except Exception as e:
            # Handle error in setting new data in the table
            print("Source: gui/windows/results_plot_window.py")
            print("Function: reinitialize_results_plot")
            print("error: " + str(e))
