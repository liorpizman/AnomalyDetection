#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Pre tune model window which is part of GUI application
'''

import os

import win32api

from gui.shared.inputs_validation_helper import pre_tune_model_path_validation
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS, clear_text, set_file_path, set_path
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


class PreTuneModel(tk.Frame):
    """
    A Class used to get the data in order to tune model parameters by user selection

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    back_window()
            Description | Handle a click on back button

    next_window()
            Description | Handle a click on next button

    reinitialize()
            Description | Reinitialize frame values and view

    input_file_browse_command()
            Description | Set the input file path to entry widget

    results_browse_command()
            Description | Set the results path to entry widget

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

        clear_text(self.path_input)

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

        if pre_tune_model_path_validation(self.path_input.get(), self.results_path_input.get()):
            self.controller.set_tune_model_input_path(self.path_input.get())
            self.controller.set_tune_model_results_path(self.results_path_input.get())
            self.controller.set_tune_model_features()
            self.controller.reinitialize_frame("TuneModel")
        else:
            win32api.MessageBox(0, 'Input path file is invalid!', 'Invalid input', 0x00001000)

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.path_label = tk.Label(self)
        self.path_label.place(relx=0.015, rely=0.4, height=32, width=146)
        self.path_label.configure(text='''Input file:''')
        set_widget_to_left(self.path_label)

        self.path_input = tk.Entry(self)
        self.path_input.place(relx=0.195, rely=0.4, height=25, relwidth=0.624)

        self.browse_btn = tk.Button(self, command=self.input_file_browse_command)
        self.browse_btn.place(relx=0.833, rely=0.4, height=25, width=60)
        set_button_configuration(self.browse_btn, text='''Browse''')

        # Results output directory
        self.results_path_label = tk.Label(self)
        self.results_path_label.place(relx=0.015, rely=0.5, height=32, width=146)
        self.results_path_label.configure(text='''Results directory:''')
        set_widget_to_left(self.results_path_label)

        self.results_path_input = tk.Entry(self)
        self.results_path_input.place(relx=0.195, rely=0.5, height=25, relwidth=0.624)

        self.results_browse_btn = tk.Button(self, command=self.results_browse_command)
        self.results_browse_btn.place(relx=0.833, rely=0.5, height=25, width=60)
        set_button_configuration(self.results_browse_btn, text='''Browse''')

    def input_file_browse_command(self):
        """
        Set the path to entry widget
        :return: updated path
        """

        self.path_input.delete(0, tk.END)
        path = set_file_path()
        self.path_input.insert(0, path)

    def results_browse_command(self):
        """
        Set the results path to entry widget
        :return: updated path
        """

        self.results_path_input.delete(0, tk.END)
        path = set_path()
        self.results_path_input.insert(0, path)
