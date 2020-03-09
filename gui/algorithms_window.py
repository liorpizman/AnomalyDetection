import time
import tkinter as tk
import tkinter.ttk as ttk
from functools import partial

import win32api

from gui.checkbox import Checkbar
from gui.utils.helper_methods import load_anomaly_detection_list
from utils.shared.input_settings import input_settings


class AlgorithmsWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # Create Widgets
        self.algorithms_title = tk.Label(self, text="Choose anomaly detection algorithms", font=controller.title_font)

        self.anomaly_detection_methods = Checkbar(self,
                                                  load_anomaly_detection_list(),
                                                  buttonCallback=self.show_algorithms_options,
                                                  editButtons=True)

        self.features_columns_options = {}

        self.menubutton = tk.Menubutton(self, text="Features",
                                   indicatoron=True, borderwidth=3, relief="raised")

        self.back_button = tk.Button(self, text="Back to menu",
                                     command=lambda: controller.show_frame("MainWindow"))

        self.next_button = tk.Button(self, text="Next",
                                     command=self.validate_next_step)

        # Layout using grid
        self.algorithms_title.grid(row=0, column=2, pady=3)
        self.anomaly_detection_methods.grid(row=2, column=2, pady=3)
        self.menubutton.grid(row=4, column=0,columnspan=150, pady=3)
        self.grid_rowconfigure(5, minsize=30)
        self.back_button.grid(row=50, column=2, pady=3)
        self.next_button.grid(row=50, column=3, pady=3)


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
        elif input_settings.get_algorithms() != set():
            self.controller.set_users_selected_features(features_list)
            self.controller.show_frame("FeatureSelectionWindow")
        else:
            win32api.MessageBox(0, 'Please edit LSTM parameters before the next step.', 'Invalid Parameters',
                                0x00001000)

    def get_features_columns_options(self):
        return self.controller.get_features_columns_options()

    def reinitialize_frame(self):
        #initialize features columns options
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


