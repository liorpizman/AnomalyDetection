#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os
import win32api

from gui.widgets.checkbox import Checkbar
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import load_anomaly_detection_list, CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_copyright_configuration, \
    set_button_configuration, set_menu_button_configuration, set_widget_to_left
from utils.input_settings import InputSettings

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

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(
            text='''Please select the algorithms for which you want to build anomaly detection models.''')
        set_widget_to_left(self.instructions)

        self.anomaly_detection_methods = Checkbar(self,
                                                  picks=load_anomaly_detection_list(),
                                                  editButtons=True,
                                                  checkCallback=self.set_algorithm_checked)
        self.anomaly_detection_methods.place(relx=0.1, rely=0.35, height=400, width=700)

        # Page footer
        self.next_button = tk.Button(self, command=self.next_window)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Next''')

        self.back_button = tk.Button(self, command=lambda: controller.show_frame("NewModel"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def show_algorithms_options(self, algorithm_name):
        self.controller.set_current_algorithm_to_edit(algorithm_name)
        self.controller.reinitialize_frame("ParametersOptionsWindow")

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.controller.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def check_algorithm_selected(self):
        for check, var in zip(self.anomaly_detection_methods.get_checks(),
                              self.anomaly_detection_methods.get_vars()):
            if var.get():
                return True
        win32api.MessageBox(0, 'Please select algorithm before the next step.', 'Invalid Algorithm',
                            0x00001000)
        return False

    def check_algorithm_parameters_edited(self):
        selected_algorithms = self.controller.get_algorithms()
        for check, var in zip(self.anomaly_detection_methods.get_checks(),
                              self.anomaly_detection_methods.get_vars()):
            current_algorithm = check.cget("text")
            algoritm_selected = var.get()
            if algoritm_selected and current_algorithm not in selected_algorithms:
                win32api.MessageBox(0, 'Please edit algorithm parameters before the next step.', 'Invalid Parameters',
                                    0x00001000)
                return False
        return True

    def validate_next_step(self):
        if self.check_algorithm_selected() and self.check_algorithm_parameters_edited():
            return True
        return False

    def next_window(self):
        if self.validate_next_step():
            self.controller.reinitialize_frame("SimilarityFunctionsWindow")

    def set_algorithm_checked(self):
        for check, var in zip(self.anomaly_detection_methods.get_checks(),
                              self.anomaly_detection_methods.get_vars()):
            if not var.get():
                self.controller.remove_algorithm(check.cget("text"))
