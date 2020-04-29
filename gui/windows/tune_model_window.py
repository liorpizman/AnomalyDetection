#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Tune model window which is part of GUI application
'''

import os

import win32api

from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS, set_path, load_anomaly_detection_list
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
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


class TuneModel(tk.Frame):
    """
    A Class used to tune model parameters by user selection

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle a click on back button

    next_window()
            Description | Handle a click on next button

    get_features_columns_options()
            Description | Get selected data set columns by the user

    reinitialize()
            Description | Reinitialize frame values and view

    get_selected_features()
            Description | Get selected features by the user

    browse_command()
            Description | Set the path to entry widget

    validate_next_step()
            Description | Validation before passing to next step

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
        self.instructions.configure(
            text='''Please select your choice for parameters tuning:''')
        set_widget_to_left(self.instructions)

        # Page body
        self.input_instructions = tk.Label(self)
        self.input_instructions.place(relx=0.05, rely=0.4, height=32, width=100)
        self.input_instructions.configure(text='''Input features:''')
        set_widget_to_left(self.input_instructions)

        self.target_instructions = tk.Label(self)
        self.target_instructions.place(relx=0.3, rely=0.4, height=32, width=100)
        self.target_instructions.configure(text='''Target features:''')
        set_widget_to_left(self.target_instructions)

        self.window_instructions = tk.Label(self)
        self.window_instructions.place(relx=0.55, rely=0.4, height=32, width=100)
        self.window_instructions.configure(text='''Window sizes:''')
        set_widget_to_left(self.window_instructions)

        self.algorithm_instructions = tk.Label(self)
        self.algorithm_instructions.place(relx=0.78, rely=0.45, height=25, width=130)
        self.algorithm_instructions.configure(text='''Algorithm:''')
        set_widget_to_left(self.algorithm_instructions)

        self.reinitialize()

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

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        self.controller.set_new_model_running(False)
        self.controller.show_frame("MainWindow")

    def next_window(self):
        """
        Handle a click on next button
        :return: if validations pass move to next window
        """

        if not self.validate_next_step():
            return
        else:
            current_features, target_features, chosen_window_sizes = self.get_selected_features()
            self.controller.set_tune_model_configuration(current_features,
                                                         target_features,
                                                         chosen_window_sizes,
                                                         self.algorithm_combo.get())
            a = self.controller.get_tune_flow_input_features()
            b = self.controller.get_tune_flow_target_features()
            c = self.controller.get_tune_flow_window_size()
            d = self.controller.get_tune_flow_algorithm()
            e = self.controller.get_tune_model_input_path()
            self.controller.reinitialize_frame("TuningLoadingWindow")

    def get_features_columns_options(self):
        """
        Get selected data set columns by the user
        :return: selected columns
        """

        return self.controller.get_tune_model_features()

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
        self.features_listbox.place(relx=0.05, rely=0.45, height=200, width=140)

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
        self.target_features_listbox.place(relx=0.3, rely=0.45, height=200, width=140)

        window_options = tk.StringVar()
        numbers = list(range(1, 16))
        window_options.set([str(i) for i in numbers])

        self.window_size_listbox = tk.Listbox(self,
                                              listvariable=window_options,
                                              selectmode=tk.MULTIPLE,
                                              exportselection=0,
                                              # Fix : ComboBox clears unrelated ListBox selection
                                              width=120,
                                              height=180,
                                              bd=3,
                                              bg='antique white',
                                              selectbackground='sandy brown')
        self.window_size_listbox.place(relx=0.55, rely=0.45, height=200, width=140)

        algorithms_list = load_anomaly_detection_list()

        self.algorithm_combo = ttk.Combobox(self, state="readonly", values=algorithms_list)
        self.algorithm_combo.place(relx=0.78, rely=0.5, height=25, width=130)
        self.algorithm_combo.current(0)

    def get_selected_features(self):
        """
        Get selected features by the user
        :return: selected features
        """

        features = []
        target_features = []
        window_sizes = []

        selection = self.features_listbox.curselection()
        target_selection = self.target_features_listbox.curselection()
        window_selection = self.window_size_listbox.curselection()

        for i in selection:
            selected = self.features_listbox.get(i)
            features.append(selected)

        for i in target_selection:
            target_selected = self.target_features_listbox.get(i)
            target_features.append(target_selected)

        for i in window_selection:
            window_selected = self.window_size_listbox.get(i)
            window_sizes.append(window_selected)

        return features, target_features, window_sizes

    def validate_next_step(self):
        """
        Validation before passing to next step
        :return: True in case validation passed, otherwise False
        """

        current_features, target_features, chosen_window_sizes = self.get_selected_features()
        if not current_features \
                or not target_features \
                or not chosen_window_sizes \
                or len(chosen_window_sizes) < 1 \
                or len(current_features) < 2 \
                or len(target_features) < 2:
            win32api.MessageBox(0,
                                'Please select at least two features for input, two features for output and window size before the next step.',
                                'Invalid Feature',
                                0x00001000)
            return False

        return True
