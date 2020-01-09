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

        self.back_button = tk.Button(self, text="Back to menu",
                                     command=lambda: controller.show_frame("MainWindow"))

        self.next_button = tk.Button(self, text="Next",
                                     command=self.validate_next_step)

        # Layout using grid
        self.algorithms_title.grid(row=0, column=2, pady=3)
        self.anomaly_detection_methods.grid(row=2, column=2, pady=3)

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
        if input_settings.get_algorithms() != set():
            self.controller.show_frame("FeatureSelectionWindow")
        else:
            win32api.MessageBox(0, 'Please edit LSTM parameters before the next step.', 'Invalid Parameters',
                                0x00001000)
