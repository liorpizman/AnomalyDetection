#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Similarity functions window which is part of GUI application
'''

import os
import win32api

from tkinter.font import Font, BOLD
from gui.widgets.checkbox import Checkbar
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import load_similarity_list, CROSS_WINDOWS_SETTINGS
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


class SimilarityFunctionsWindow(tk.Frame):
    """
    A Class used to enable the user to choose a similarity functions

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    set_similarity_score()
            Description | Set selected similarity functions in global input settings

    next_window()
            Description | Handle a click on next button

    back_window()
            Description | Handle a click on back button

    reinitialize()
            Description | Reinitialize frame values and view

    set_saving_model()
            Description | Set indicator whether the user want to save the model or not

    similarity_functions_validation()
            Description | Validations that at least one similarity function was checked

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
            text='''Please choose similarity functions from the following options.''')
        set_widget_to_left(self.instructions)

        # Page body
        self.similarity_functions = Checkbar(self, load_similarity_list(), checkCallback=self.set_similarity_score)
        self.similarity_functions.place(relx=0.1, rely=0.36, height=400, width=700)

        self.save_model_var = tk.IntVar()
        self.save_model_check_button = tk.Checkbutton(self,
                                                      text="Save model",
                                                      variable=self.save_model_var,
                                                      command=self.set_saving_model)

        self.note = tk.Label(self)
        self.note.place(relx=0.015, rely=0.7, height=32, width=635)
        self.note.configure(
            text='''Note: Similarity function is used for calculating a score for each record''',
            font=Font(size=9, weight=BOLD))
        set_widget_to_left(self.note)

        # Page footer
        self.next_button = tk.Button(self, command=self.next_window)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Run''')

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

        for check, var in zip(self.similarity_functions.get_checks(),
                              self.similarity_functions.get_vars()):
            var.set(0)
            check['variable'] = var
            check['state'] = 'active'

        # In case the check button was not destroyed - should be in new model flow
        is_new_model_flow = self.controller.get_new_model_running()

        if is_new_model_flow:
            self.save_model_var.set(0)
            self.save_model_check_button['variable'] = self.save_model_var

    def set_similarity_score(self):
        """
        Set selected similarity functions in global input settings
        :return: updated global settings
        """

        similarity_list = set()

        for check, var in zip(self.similarity_functions.get_checks(),
                              self.similarity_functions.get_vars()):
            if var.get():
                similarity_list.add(check.cget("text"))
        self.controller.set_similarity_score(similarity_list)

    def next_window(self):
        """
        Handle a click on next button
        :return: if validations pass move to next window
        """

        if self.similarity_functions_validation():
            self.controller.reinitialize_frame("LoadingWindow")
        else:
            win32api.MessageBox(0, 'Please select at least one similarity function', 'Invalid input', 0x00001000)

    def back_window(self):
        """
        Handle back button click
        :return: previous window
        """

        is_new_model_flow = self.controller.get_new_model_running()

        if is_new_model_flow:
            self.controller.show_frame("FeatureSelectionWindow")
        else:
            self.controller.show_frame("ExistingAlgorithmsWindow")

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        if self.controller.get_new_model_running():
            if not self.save_model_var:
                self.save_model_var = tk.IntVar()
            if not self.save_model_check_button:
                self.save_model_check_button = tk.Checkbutton(self,
                                                              text="Save model",
                                                              variable=self.save_model_var,
                                                              command=self.set_saving_model)
            self.save_model_check_button.place(relx=0.65, rely=0.75, height=25, width=100)
        else:
            self.save_model_check_button.place_forget()

    def set_saving_model(self):
        """
        Set indicator whether the user want to save the model or not
        :return: updated indicator
        """

        self.controller.set_saving_model(self.save_model_var.get() == 1)

    def similarity_functions_validation(self):
        """
        Validations that at least one similarity function was checked
        :return: True if at least one was checked, otherwise false
        """

        for var in self.similarity_functions.get_vars():
            if var.get() != 0:
                return True

        return False
