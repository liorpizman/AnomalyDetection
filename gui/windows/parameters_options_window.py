#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Parameters options window which is part of GUI application
'''

import os

from gui.algorithm_frame_options.algorithm_frame_options import AlgorithmFrameOptions
from gui.widgets.hover_button import HoverButton
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.shared.helper_methods import load_anomaly_detection_list
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
    set_button_configuration, set_copyright_configuration

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


class ParametersOptionsWindow(tk.Frame):
    """
    A Class used to show all the parameters for each algorithm in the application

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    handle_next_button()
            Description | Handle a click on next button

    set_algorithm_parameters(algorithm_name, algorithm_parameters)
            Description | Set parameters for each algorithm

    save_algorithm_parameters(algorithm_parameters)
            Description | Save the parameters for each algorithm which was chosen

    set_suitable_yaml_file(algorithm_name)
            Description | Set the yaml file according to algorithm name

    reinitialize()
            Description | Reinitialize frame values and view

    reinitialize_current_algorithm_options()
            Description |  Reinitialize algorithm value and view

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
            text='''Please select the values for each of the following parameters:''')
        set_widget_to_left(self.instructions)

        # Page body

        # Dynamic algorithm options
        self.algorithms_files = load_anomaly_detection_list()
        self.current_algorithm = self.controller.get_current_algorithm_to_edit()
        self.current_yaml = self.set_suitable_yaml_file(self.current_algorithm)

        self.height_options_frame = 268
        self.width_options_frame = 620

        self.options_to_show = AlgorithmFrameOptions(self, yaml_filename=self.current_yaml)
        self.options_to_show.place(relx=0.05,
                                   rely=0.35,
                                   height=self.height_options_frame,
                                   width=self.width_options_frame)

        # Page footer
        self.next_button = HoverButton(self, command=self.handle_next_button)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Save''')

        self.back_button = HoverButton(self, command=lambda: self.controller.show_frame("AlgorithmsWindow"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Cancel''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def reset_widgets(self):
        """
        Reset check bar values
        :return: empty values in the widgets
        """

        pass

    def handle_next_button(self):
        """
        Handle a click on next button
        :return: if validations pass move to next window
        """

        algorithm_parameters = self.options_to_show.get_algorithm_parameters()
        self.save_algorithm_parameters(algorithm_parameters)

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        """
        Set parameters for each algorithm
        :param algorithm_name: the name of the algorithm
        :param algorithm_parameters: the values of the parameteres
        :return: updates state
        """

        self.controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def save_algorithm_parameters(self, algorithm_parameters):
        """
        Save the parameters for each algorithm which was chosen
        :param algorithm_parameters: new values
        :return: updated state of user choice
        """

        algorithm_name = self.controller.get_current_algorithm_to_edit()
        self.set_algorithm_parameters(algorithm_name, algorithm_parameters)
        self.controller.show_frame("AlgorithmsWindow")

    def set_suitable_yaml_file(self, algorithm_name):
        """
        Set the yaml file according to algorithm name
        :param algorithm_name: the name of the algorithm
        :return: yaml file
        """

        switcher = {
            self.algorithms_files[0]: "lstm_params.yaml",
            self.algorithms_files[1]: "svr_params.yaml",
            self.algorithms_files[2]: "mlp_params.yaml",
            self.algorithms_files[3]: "random_forest_params.yaml",
        }

        return switcher.get(algorithm_name, None)

    def reinitialize(self):
        """
        Reinitialize frame values and view
        :return: new frame view
        """

        self.reinitialize_current_algorithm_options()

    def reinitialize_current_algorithm_options(self):
        """
        Reinitialize algorithm value and view
        :return: new frame view
        """

        self.current_algorithm = self.controller.get_current_algorithm_to_edit()
        self.current_yaml = self.set_suitable_yaml_file(self.current_algorithm)
        self.controller.remove_algorithm(self.current_algorithm)

        self.options_to_show.destroy()
        self.options_to_show = AlgorithmFrameOptions(self, yaml_filename=self.current_yaml)
        self.options_to_show.place(relx=0.05, rely=0.35, height=268, width=620)
