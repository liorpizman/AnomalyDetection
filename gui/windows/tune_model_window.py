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

        pass

    def get_features_columns_options(self):
        """
        Get selected data set columns by the user
        :return: selected columns
        """

        return self.controller.get_features_columns_options()

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.path_label = tk.Label(self)
        self.path_label.place(relx=0.015, rely=0.35, height=32, width=146)
        self.path_label.configure(text='''Input directory:''')
        set_widget_to_left(self.path_label)

        self.path_input = tk.Entry(self)
        self.path_input.place(relx=0.195, rely=0.35, height=25, relwidth=0.624)

        self.browse_btn = tk.Button(self, command=self.browse_command)
        self.browse_btn.place(relx=0.833, rely=0.35, height=25, width=60)
        set_button_configuration(self.browse_btn, text='''Browse''')

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

        self.window_size_listbox = tk.Listbox(self,
                                              listvariable=tk.StringVar().set(
                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
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
        window_selectioo = self.window_size_listbox.curselection()

        for i in selection:
            selected = self.features_listbox.get(i)
            features.append(selected)

        for i in target_selection:
            target_selected = self.target_features_listbox.get(i)
            target_features.append(target_selected)

        for i in window_selectioo:
            window_selected = self.window_size_listbox.get(i)
            window_sizes.append(window_selected)

        return features, target_features, window_sizes

    def browse_command(self):
        """
        Set the path to entry widget
        :return: updated path
        """

        self.path_input.delete(0, tk.END)
        path = set_path()
        self.path_input.insert(0, path)
