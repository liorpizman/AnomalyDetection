#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Check bar which is presented in the application
'''

from tkinter import *

from gui.shared.helper_methods import load_anomaly_detection_list
from gui.widgets_configurations.helper_methods import set_widget_to_left


class Checkbar(Frame):
    """
     A Class used to show paris of check buttons and buttons

    Methods
    -------
    state()
            Description | Get current state of the pairs

    get_checkbar_state()
            Description | Get the state of the check bar

    get_checks()
            Description | Get the values of the checks

    get_vars()
            Description | Get the values of the variables

    get_buttons()
            Description | Get the buttons widgets

    set_button_state(checkCallback)
            Description | Set the state of all the button according to the variables

    show_LSTM_options()
            Description | Callback to open LSTM parameters options

    show_SVR_options()
            Description | Callback to open SVR parameters options

    show_MLP_options()
            Description | Callback to open MLP parameters options

    show_Random_Forest_options()
            Description | Callback to open Random forest parameters options

    get_algorithm_show_function(algorithm_name)
            Description | Switch to get the callback according to a given algorithm name

    """

    def __init__(self, parent=None, picks=[], editButtons=False, checkCallback=None):
        """
        Parameters
        ----------

        :param parent: the parent window
        :param picks: the list of pairs to show
        :param editButtons: Indicator whether a configuration button should be displayed
        :param checkCallback: Callback for each pair
        """
        Frame.__init__(self, parent)

        self.parent = parent
        self.vars = []
        self.checks = []
        self.buttons = []

        relY = 0
        relX = 0
        enable_functionality = 'active'

        # Iterate over all pairs of check buttons and buttons
        for pick in picks:
            var = IntVar()

            algorithm_show_function = self.get_algorithm_show_function(str(pick))

            # Create a check button dynamically
            check_button = Checkbutton(self,
                                       text=pick,
                                       variable=var,
                                       state=enable_functionality,
                                       command=lambda: self.set_button_state(checkCallback))

            check_button.place(relx=relX, rely=relY, height=30, width=150)
            set_widget_to_left(check_button)

            if editButtons:
                # Create a configuration button dynamically
                edit_button = Button(self,
                                     text=pick + " configuration",
                                     state='disabled',
                                     command=algorithm_show_function)

                edit_button.place(relx=relX + 0.35, rely=relY, height=30, width=220)
                self.buttons.append(edit_button)
            self.vars.append(var)
            self.checks.append(check_button)
            relY = relY + 0.1

    def state(self):
        """
        Get current state of the pairs
        :return: map of pair values
        """

        return map((lambda var: var.get()), self.vars)

    def get_checkbar_state(self):
        """
        Get the state of the check bar
        :return: values of the checks and the vars
        """

        return self.checks, self.vars

    def get_checks(self):
        """
        Get the values of the checks
        :return: checks' values
        """

        return self.checks

    def get_vars(self):
        """
        Get the values of the variables
        :return: variables' values
        """

        return self.vars

    def get_buttons(self):
        """
        Get the buttons widgets
        :return:
        """

        return self.buttons

    def set_button_state(self, checkCallback):
        """
        Set the state of all the button according to the variables
        :param checkCallback: on click callback
        :return: buttons in updated state
        """
        for button, var in zip(self.buttons, self.vars):
            if var.get():
                button['state'] = 'active'
            else:
                button['state'] = 'disabled'
        if checkCallback is not None:
            checkCallback()

    def show_LSTM_options(self):
        """
        Callback to open LSTM parameters options
        :return: suitable parameters for LSTM algorithm
        """

        self.parent.show_algorithms_options(load_anomaly_detection_list()[0])

    def show_SVR_options(self):
        """
        Callback to open SVR parameters options
        :return: suitable parameters for SVR algorithm
        """

        self.parent.show_algorithms_options(load_anomaly_detection_list()[1])

    def show_MLP_options(self):
        """
        Callback to open MLP parameters options
        :return: suitable parameters for MLP algorithm
        """

        self.parent.show_algorithms_options(load_anomaly_detection_list()[2])

    def show_Random_Forest_options(self):
        """
        Callback to open Random forest parameters options
        :return: suitable parameters for Random forest algorithm
        """

        self.parent.show_algorithms_options(load_anomaly_detection_list()[3])

    def get_algorithm_show_function(self, algorithm_name):
        """
        Switch to get the callback according to a given algorithm name
        :param algorithm_name: input algorithm
        :return: callback suitable for input algorithm name
        """

        algorithms = load_anomaly_detection_list()

        # Switch between algorithms' callbacks
        switcher = {
            algorithms[0]: self.show_LSTM_options,
            algorithms[1]: self.show_SVR_options,
            algorithms[2]: self.show_MLP_options,
            algorithms[3]: self.show_Random_Forest_options
        }

        return switcher.get(algorithm_name, None)
