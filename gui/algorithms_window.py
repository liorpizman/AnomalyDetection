#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os
import win32api

from gui.checkbox import Checkbar
from gui.menubar import Menubar
from gui.utils.helper_methods import load_anomaly_detection_list, CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_copyright_configuration, \
    set_button_configuration, set_menu_button_configuration, set_widget_to_left
from utils.shared.input_settings import InputSettings

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


class AlgorithmsWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.menubar = Menubar(controller)
        self.controller.option_add('*tearOff', 'FALSE')  # Disables ability to tear menu bar into own window
        system_logo = CROSS_WINDOWS_SETTINGS.get('LOGO')
        photo_location = os.path.join(system_logo)
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(
            text='''Please select the algorithms for which you want to build anomaly detection models.''')
        set_widget_to_left(self.instructions)

        self.anomaly_detection_methods = Checkbar(self,
                                                  load_anomaly_detection_list(),
                                                  buttonCallback=self.show_algorithms_options,
                                                  editButtons=True)
        self.anomaly_detection_methods.place(relx=0.1, rely=0.35, height=400, width=700)
        self.features_columns_options = {}

        self.menubutton = tk.Menubutton(self)
        self.menubutton.place(relx=0.1, rely=0.65, height=25, width=81)
        set_menu_button_configuration(self.menubutton)

        self.next_button = tk.Button(self, command=self.validate_next_step)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Next''')

        self.back_button = tk.Button(self, command=lambda: controller.show_frame("NewModel"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def show_algorithms_options(self):
        for check, var in zip(self.anomaly_detection_methods.get_checks(),
                              self.anomaly_detection_methods.get_vars()):
            algorithm_name = check.cget("text")
            if algorithm_name != "LSTM":
                continue
            self.controller.show_frame("LSTMWindow")

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.controller.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def validate_next_step(self):
        features_list = self.get_selected_features()
        if not features_list:
            win32api.MessageBox(0, 'Please select feature for the models before the next step.', 'Invalid Feature',
                                0x00001000)
        elif InputSettings.get_algorithms() != set():
            self.controller.set_users_selected_features(features_list)
            self.controller.show_frame("SimilarityFunctionsWindow")
        else:
            win32api.MessageBox(0, 'Please edit LSTM parameters before the next step.', 'Invalid Parameters',
                                0x00001000)

    def get_features_columns_options(self):
        return self.controller.get_features_columns_options()

    def reinitialize(self):
        # initialize features columns options
        self.features_columns_options = {}
        self.menu = tk.Menu(self.menubutton, tearoff=False)
        self.menubutton.configure(menu=self.menu)
        for feature in self.get_features_columns_options():
            self.features_columns_options[feature] = tk.IntVar(value=0)
            self.menu.add_checkbutton(label=feature, variable=self.features_columns_options[feature],
                                      onvalue=1, offvalue=0)

    def get_selected_features(self):
        features = []
        for name, var in self.features_columns_options.items():
            if var.get():
                features.append(name)
        return features
