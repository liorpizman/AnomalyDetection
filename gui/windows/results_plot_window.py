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
import shutil

from datetime import datetime
from tkinter import messagebox
from tkinter.font import Font, BOLD
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import trim_unnecessary_chars, set_path
from gui.widgets.hover_button import HoverButton
from gui.widgets.menubar import Menubar

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
    A Class used to present a results plots permutation by an algorithm and a flight route

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

    show_plot(img_url)
            Description | Show a chosen plot

    export_plot_to_png(algorithm, similarity_function, flight_route)
            Description | Export current plot to png file

    parseAttacksFromPaths(paths_list)
            Description | Parse attacks names

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
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

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

            current_title = 'Test set attacks plots:'

            self.instructions = tk.Label(self)
            self.instructions.place(relx=0.015, rely=0.29, height=35, width=635)
            self.instructions.configure(text=current_title)
            set_widget_to_left(self.instructions)

            # Page header
            plot_key = get_plots_key(original_algorithm, original_similarity_function, original_flight_route)
            self.plots_list = InputSettings.get_plots(plot_key)
            self.plots_index = 0

            start_y = 0.33
            self.plot_buttons = []
            self.export_buttons = []
            print(self.plots_list)

            attacks = self.parseAttacksFromPaths(self.plots_list)

            for attack, image_plot_path in zip(attacks, self.plots_list):
                self.plot_label = tk.Label(self)
                self.plot_label.place(relx=0.015, rely=start_y, height=35, width=180)
                self.plot_label.configure(text='To view {0} graph'.format(attack))
                set_widget_to_left(self.plot_label)

                self.plot_button = tk.Button(self, command=lambda plot_path=image_plot_path: self.show_plot(plot_path))
                self.plot_button.place(relx=0.315, rely=start_y, height=25, width=72)
                set_button_configuration(self.plot_button, text='''Click here''')
                self.plot_button.configure(bg='navajo white')

                self.export_button = tk.Button(self,
                                               command=lambda plot_path=image_plot_path: self.export_plot_to_png(
                                                   algorithm=selected_algorithm,
                                                   similarity_function=selected_similarity_function,
                                                   flight_route=selected_flight_route,
                                                   img_url=plot_path
                                               ))
                self.export_button.place(relx=0.485, rely=start_y, height=25, width=90)
                set_button_configuration(self.export_button, text='''Export to PNG''')
                self.export_button.configure(bg='sandy brown')

                start_y += 0.08

            permutation_styling = Font(family="Times New Roman",
                                       size=11,
                                       weight=BOLD)

            self.algorithm_label = tk.Label(self)
            self.algorithm_label.place(relx=0.015, rely=0.68, height=25, width=300)
            self.algorithm_label.configure(
                text="Algorithm: {0}".format(selected_algorithm),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.algorithm_label)

            self.similarity_function_label = tk.Label(self)
            self.similarity_function_label.place(relx=0.015, rely=0.72, height=25, width=300)
            self.similarity_function_label.configure(
                text="Similarity function: {0}".format(selected_similarity_function),
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.similarity_function_label)

            self.route_label = tk.Label(self)
            self.route_label.place(relx=0.015, rely=0.76, height=25, width=300)
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

    def show_plot(self, img_url):
        """
        Show a chosen plot
        :return: pop up plot
        """

        novi = tk.Toplevel()
        canvas = tk.Canvas(novi, width=1450, height=360)
        canvas.pack(expand=YES, fill=BOTH)
        gif1 = tk.PhotoImage(file=img_url)
        plot_img = gif1.subsample(2)
        canvas.create_image(10, 10, image=plot_img, anchor=NW)
        canvas.gif1 = plot_img

    def export_plot_to_png(self, algorithm, similarity_function, flight_route, img_url):
        """
        Export current plot to png file
        :param algorithm: chosen algorithm
        :param similarity_function: chosen similarity function
        :param flight_route: chosen flight route
        :param img_url: path of the plot
        :return: exported png
        """

        path = set_path()
        current_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
        df_name = "Plot_{0}_{1}_{2}_{3}.png".format(algorithm, similarity_function, flight_route, current_time)
        full_path = os.path.join(*[str(path), df_name])

        try:
            shutil.copy(src=img_url, dst=full_path)

            messagebox.askokcancel(
                title="Export to PNG file",
                message="Export to PNG file finished successfully!"
            )
        except:
            messagebox.askokcancel(
                title="Export to PNG file",
                message="Export to PNG file failed! Please try again."
            )

    def parseAttacksFromPaths(self, paths_list):
        """
        Parse attacks names
        :param paths_list: full paths for each png
        :return: attacks
        """

        attacks = []

        for path in paths_list:
            split_by_bracket_opener = path.split('(')
            split_by_bracket_closer = split_by_bracket_opener[len(split_by_bracket_opener) - 1].split(')')[0]
            attacks.append(split_by_bracket_closer.replace('_', ' '))

        return attacks
