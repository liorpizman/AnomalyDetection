#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Display loaded data window which is part of GUI application
'''

import os
from tkinter import messagebox
from tkinter.font import Font, BOLD

from gui.widgets.hover_button import HoverButton
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS, read_json_file
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


class LoadedDataWindow(tk.Frame):
    """
    A Class used to display hyper parameters

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle back button click

    next_window()
            Description | Handle next button click

    reinitialize()
            Description | Reinitialize frame values and view

    show_features(data)
            Description | Show all features

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
        self.instructions.configure(text='''Hyper parameters from existing models:''')
        set_widget_to_left(self.instructions)

        # Page body

        # Page footer
        self.next_button = HoverButton(self, command=self.next_window)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Next''')

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

        self.controller.show_frame("ExistingAlgorithmsWindow")

    def next_window(self):
        """
        Handle next button click
        :return: if validations pass move to next window
        """

        self.controller.reinitialize_frame("SimilarityFunctionsWindow")

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.model_data_algorithms = self.controller.get_existing_algorithms()
        self.algorithms_display_data = dict()

        for algorithm, path in self.model_data_algorithms.items():
            full_path = os.path.join(*[str(path), 'model_data.json'])
            self.algorithms_display_data[algorithm] = read_json_file(full_path)

        permutation_styling = Font(family="Times New Roman",
                                   size=9,
                                   weight=BOLD)

        relative_x = 0.01
        height = 25
        width = 160

        for alogrithm, data in self.algorithms_display_data.items():
            relative_y = 0.34

            self.algorithm_label = tk.Label(self)
            self.algorithm_label.place(relx=relative_x, rely=relative_y, height=height, width=width)
            self.algorithm_label.configure(
                text=alogrithm,
                font=permutation_styling,
                fg='blue')
            set_widget_to_left(self.algorithm_label)

            relative_y += 0.03

            self.features_label = tk.Label(self)
            self.features_label.place(relx=relative_x, rely=relative_y, height=height, width=width)
            self.features_label.configure(text="Input features: ")
            set_widget_to_left(self.features_label)

            self.input_features_button = tk.Button(self,
                                                   command=lambda: self.show_features(
                                                       title="Input features window",
                                                       data=data["features"]
                                                   ))
            self.input_features_button.place(relx=relative_x + 0.13, rely=relative_y, height=15, width=55)
            set_button_configuration(self.input_features_button, text='''Show all''')
            self.input_features_button.configure(bg='sandy brown')

            relative_y += 0.035

            self.target_features_label = tk.Label(self)
            self.target_features_label.place(relx=relative_x, rely=relative_y, height=height, width=width)
            self.target_features_label.configure(text="Target features: ")
            set_widget_to_left(self.target_features_label)

            self.target_features_button = tk.Button(self,
                                                    command=lambda: self.show_features(
                                                        title="Target features window",
                                                        data=data["target_features"]
                                                    ))
            self.target_features_button.place(relx=relative_x + 0.13, rely=relative_y, height=15, width=55)
            set_button_configuration(self.target_features_button, text='''Show all''')
            self.target_features_button.configure(bg='sandy brown')

            relative_y += 0.06

            self.threshold = tk.Label(self)
            self.threshold.place(relx=relative_x, rely=relative_y, height=height, width=width)
            self.threshold.configure(text="Calculated threshold: {0}".format(format(data["threshold"], '.5f')))
            set_widget_to_left(self.threshold)

            relative_y += 0.06

            self.hyper_params_label = tk.Label(self)
            self.hyper_params_label.place(relx=relative_x, rely=relative_y, height=height, width=width)
            self.hyper_params_label.configure(
                text="Hyper params:",
                font=permutation_styling,
                fg='green')
            set_widget_to_left(self.hyper_params_label)

            relative_y += 0.04
            params = data["params"]

            for attribute in params.keys():
                self.attribute_label = tk.Label(self)
                self.attribute_label.place(relx=relative_x, rely=relative_y, height=height, width=width)
                self.attribute_label.configure(
                    text="{0}: {1}".format(attribute, params[attribute]))
                set_widget_to_left(self.attribute_label)

                relative_y += 0.03

            relative_x += 0.24

    def show_features(self, title, data):
        """
        Show all features
        :return: model's selected features
        """

        msg = ''
        for feature in data:
            msg += feature + '\n'

        messagebox.askokcancel(
            title=title,
            message=msg
        )
