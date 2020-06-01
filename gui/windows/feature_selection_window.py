#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Feature selection window which is part of GUI application
'''

import os
import win32api

from tkinter.font import Font, BOLD
from tkinter import font as tkfont

from gui.widgets.hover_button import HoverButton
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_widget_to_left, set_logo_configuration, \
    set_copyright_configuration, set_button_configuration

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


class FeatureSelectionWindow(tk.Frame):
    """
    A Class used to enable the user to select features

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    next_window()
            Description | Handle a click on next button

    back_window()
            Description | Handle a click on back button

    reinitialize()
            Description | Reinitialize frame values and view

    get_selected_features()
        Description | Get selected features by the user

    get_features_columns_options()
            Description | Get selected data set columns by the user

    validate_next_step()
        Description | Validation before passing to next step

    set_users_selected_features(features_list)
            Description | Set the selected features from the test data set

    select_all_features
            Description | Select/Clear all input features

    select_all_targets
            Description | Select/Clear all target features

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
        self.instructions.configure(text='''[Step 3/5] Please choose both input and target features:''')
        set_widget_to_left(self.instructions)

        # Page body

        # initialize features columns options
        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()

        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.features_instructions = tk.Label(self)
        self.features_instructions.place(relx=0.05, rely=0.34, height=22, width=100)
        self.features_instructions.configure(text='''Input features:''')
        set_widget_to_left(self.features_instructions)

        self.features_listbox = tk.Listbox(self,
                                           font=tkfont.Font(size=9),
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           exportselection=0,  # Fix : ComboBox clears unrelated ListBox selection
                                           width=120,
                                           height=180,
                                           bd=3,
                                           bg='antique white',
                                           selectbackground='sandy brown')
        self.features_listbox.place(relx=0.05, rely=0.42, height=230, width=140)

        self.target_instructions = tk.Label(self)
        self.target_instructions.place(relx=0.35, rely=0.34, height=22, width=100)
        self.target_instructions.configure(text='''Target features:''')
        set_widget_to_left(self.target_instructions)

        self.target_features_listbox = tk.Listbox(self,
                                                  font=tkfont.Font(size=9),
                                                  listvariable=self.csv_features,
                                                  selectmode=tk.MULTIPLE,
                                                  exportselection=0,
                                                  # Fix : ComboBox clears unrelated ListBox selection
                                                  width=120,
                                                  height=180,
                                                  bd=3,
                                                  bg='antique white',
                                                  selectbackground='sandy brown')
        self.target_features_listbox.place(relx=0.35, rely=0.42, height=230, width=140)

        # Side logo
        feature_selection_logo = CROSS_WINDOWS_SETTINGS.get('FEATURE_SELECTION')
        feature_selection_photo_location = os.path.join(feature_selection_logo)
        global fs_logo_img
        fs_logo_img = tk.PhotoImage(file=feature_selection_photo_location)

        self.features_logo_png = tk.Button(self)
        self.features_logo_png.place(relx=0.6, rely=0.28, height=200, width=200)
        set_logo_configuration(self.features_logo_png, image=fs_logo_img)

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

        self.features_listbox.selection_clear(0, tk.END)
        self.target_features_listbox.selection_clear(0, tk.END)

    def next_window(self):
        """
        Handle a click on next button
        :return: if validations pass move to next window
        """

        if not self.validate_next_step():
            return
        else:
            current_features, target_features = self.get_selected_features()
            self.set_users_selected_features(current_features, target_features)
            self.controller.reinitialize_frame("SimilarityFunctionsWindow")

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.show_frame("AlgorithmsWindow")

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()

        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.input_indicator = False
        self.target_indicator = False

        self.select_all_features_button = tk.Button(self, command=self.select_all_features)
        self.select_all_features_button.place(relx=0.17, rely=0.38, height=18, width=55)
        set_button_configuration(self.select_all_features_button, text='''Select all''')
        self.select_all_features_button.configure(bg='sandy brown')

        self.features_listbox = tk.Listbox(self,
                                           font=tkfont.Font(size=9),
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           exportselection=0,  # Fix : ComboBox clears unrelated ListBox selection
                                           width=120,
                                           height=180,
                                           bd=3,
                                           bg='antique white',
                                           selectbackground='sandy brown')
        self.features_listbox.place(relx=0.05, rely=0.42, height=230, width=140)

        self.select_all_target_button = tk.Button(self, command=self.select_all_target)
        self.select_all_target_button.place(relx=0.47, rely=0.38, height=18, width=55)
        set_button_configuration(self.select_all_target_button, text='''Select all''')
        self.select_all_target_button.configure(bg='sandy brown')

        self.target_features_listbox = tk.Listbox(self,
                                                  listvariable=self.csv_features,
                                                  font=tkfont.Font(size=9),
                                                  selectmode=tk.MULTIPLE,
                                                  exportselection=0,
                                                  # Fix : ComboBox clears unrelated ListBox selection
                                                  width=120,
                                                  height=180,
                                                  bd=3,
                                                  bg='antique white',
                                                  selectbackground='sandy brown')
        self.target_features_listbox.place(relx=0.35, rely=0.42, height=230, width=140)

        title_styling = Font(size=11,
                             weight=BOLD)

        self.previous_choice_label = tk.Label(self)
        self.previous_choice_label.place(relx=0.58, rely=0.62, height=25, width=300)
        self.previous_choice_label.configure(
            text="Your previous selections:",
            font=title_styling,
            fg='blue')
        set_widget_to_left(self.previous_choice_label)

        chosen_algorithms = self.controller.get_algorithms()

        y_coordinate = 0.66
        for algorithm in chosen_algorithms:
            window_size = self.controller.get_window_size(algorithm)

            self.algorithm_label = tk.Label(self)
            self.algorithm_label.place(relx=0.58, rely=y_coordinate, height=25, width=300)
            self.algorithm_label.configure(
                text="{0} window size: {1}".format(algorithm, window_size),
                font=Font(size=10),
                fg='blue')
            set_widget_to_left(self.algorithm_label)

            y_coordinate += 0.04

    def get_selected_features(self):
        """
        Get selected features by the user
        :return: selected features
        """

        features = []
        target_features = []

        selection = self.features_listbox.curselection()
        target_selection = self.target_features_listbox.curselection()

        for i in selection:
            selected = self.features_listbox.get(i)
            features.append(selected)

        for i in target_selection:
            target_selected = self.target_features_listbox.get(i)
            target_features.append(target_selected)

        return features, target_features

    def get_features_columns_options(self):
        """
        Get selected data set columns by the user
        :return: selected columns
        """

        return self.controller.get_features_columns_options()

    def validate_next_step(self):
        """
        Validation before passing to next step
        :return: True in case validation passed, otherwise False
        """

        current_features, target_features = self.get_selected_features()
        if not current_features or not target_features or len(current_features) < 2 or len(target_features) < 2:
            win32api.MessageBox(0,
                                'Please select at least two features for input and two features for output before the next step.',
                                'Invalid Feature',
                                0x00001000)
            return False

        return True

    def set_users_selected_features(self, features_list, target_features_list):
        """
        Set the selected features from the test data set
        :param features_list: the list of selected features for input
        :param target_features_list: the list of selected features for target
        :return: updates state of features selection
        """

        self.controller.set_users_selected_features(features_list, target_features_list)

    def select_all_features(self):
        """
        Select/Clear all input features
        :return: selected/cleared listbox
        """

        if self.input_indicator:
            self.features_listbox.selection_clear(0, tk.END)
            self.select_all_features_button.configure(bg='sandy brown', text='''Select all''')
        else:
            self.features_listbox.select_set(0, tk.END)
            self.select_all_features_button.configure(bg='firebrick1', text='''Clear all''')

        self.input_indicator = not self.input_indicator

    def select_all_target(self):
        """
        Select/Clear all target features
        :return: selected/cleared listbox
        """

        if self.target_indicator:
            self.target_features_listbox.selection_clear(0, tk.END)
            self.select_all_target_button.configure(bg='sandy brown', text='''Select all''')
        else:
            self.target_features_listbox.select_set(0, tk.END)
            self.select_all_target_button.configure(bg='firebrick1', text='''Clear all''')

        self.target_indicator = not self.target_indicator
