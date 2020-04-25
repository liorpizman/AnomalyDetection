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
        self.instructions.configure(text='''Please choose both input and target features:''')
        set_widget_to_left(self.instructions)

        # Page body

        # initialize features columns options
        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()

        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.05, rely=0.35, height=32, width=100)
        self.instructions.configure(text='''Input features:''')
        set_widget_to_left(self.instructions)

        self.features_listbox = tk.Listbox(self,
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           exportselection=0,  # Fix : ComboBox clears unrelated ListBox selection
                                           width=120,
                                           height=180,
                                           bd=3,
                                           bg='antique white',
                                           selectbackground='sandy brown')
        self.features_listbox.place(relx=0.05, rely=0.4, height=230, width=140)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.35, rely=0.35, height=32, width=100)
        self.instructions.configure(text='''Target features:''')
        set_widget_to_left(self.instructions)

        self.target_features_listbox = tk.Listbox(self,
                                                  listvariable=self.csv_features,
                                                  selectmode=tk.MULTIPLE,
                                                  exportselection=0,
                                                  # Fix : ComboBox clears unrelated ListBox selection
                                                  width=120,
                                                  height=180,
                                                  bd=3,
                                                  bg='antique white',
                                                  selectbackground='sandy brown')
        self.target_features_listbox.place(relx=0.35, rely=0.4, height=230, width=140)

        # Side logo
        feature_selection_logo = CROSS_WINDOWS_SETTINGS.get('FEATURE_SELECTION')
        feature_selection_photo_location = os.path.join(feature_selection_logo)
        global fs_logo_img
        fs_logo_img = tk.PhotoImage(file=feature_selection_photo_location)

        self.features_logo_png = tk.Button(self)
        self.features_logo_png.place(relx=0.6, rely=0.35, height=200, width=200)
        set_logo_configuration(self.features_logo_png, image=fs_logo_img)

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

        pass

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

        self.controller.show_frame("ParametersOptionsWindow")

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.features_columns_options = {}
        self.features_columns_options = self.get_features_columns_options()

        self.csv_features = tk.StringVar()
        self.csv_features.set(self.features_columns_options)

        self.features_listbox = tk.Listbox(self,
                                           listvariable=self.csv_features,
                                           selectmode=tk.MULTIPLE,
                                           exportselection=0,  # Fix : ComboBox clears unrelated ListBox selection
                                           width=120,
                                           height=180,
                                           bd=3,
                                           bg='antique white',
                                           selectbackground='sandy brown')
        self.features_listbox.place(relx=0.05, rely=0.4, height=230, width=140)

        self.target_features_listbox = tk.Listbox(self,
                                                  listvariable=self.csv_features,
                                                  selectmode=tk.MULTIPLE,
                                                  exportselection=0,
                                                  # Fix : ComboBox clears unrelated ListBox selection
                                                  width=120,
                                                  height=180,
                                                  bd=3,
                                                  bg='antique white',
                                                  selectbackground='sandy brown')
        self.target_features_listbox.place(relx=0.35, rely=0.4, height=230, width=140)

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
