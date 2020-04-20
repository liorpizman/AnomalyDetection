#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Main window which is part of GUI application
'''

import os

from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_button_configuration, set_logo_configuration, \
    set_copyright_configuration

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


class MainWindow(tk.Frame):
    """
    A Class used to configure and populate the main window

    Methods
    -------
    reset_widgets()
            Description | Reset check bar values

    load_flow()
            Description | Move to load model flow

    new_flow()
            Description | Move to new model flow

    """

    def __init__(self, parent, controller):
        """
        Parameters
        ----------

        :param parent: window
        :param controller: GUI controller
        """

        ttk.Frame.__init__(self, parent)

        # Page init
        self.controller = controller
        self.menubar = Menubar(controller)
        # Disables ability to tear menu bar into own window
        self.controller.option_add('*tearOff', 'FALSE')
        system_logo = CROSS_WINDOWS_SETTINGS.get('LOGO')
        photo_location = os.path.join(system_logo)
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        self.controller.geometry("700x550")
        self.controller.minsize(700, 550)
        self.controller.maxsize(700, 550)
        self.controller.resizable(1, 1)
        self.controller.title("Anomaly Detection System")
        self.controller.configure(background="#eeeeee")

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        # Page body
        self.create_model_btn = tk.Button(self, command=self.new_flow)
        self.create_model_btn.place(relx=0.41, rely=0.5, height=42, width=120)
        set_button_configuration(self.create_model_btn, text='''Create model''')

        self.load_model_btn = tk.Button(self, command=self.load_flow)
        self.load_model_btn.place(relx=0.41, rely=0.7, height=42, width=120)
        set_button_configuration(self.load_model_btn, text='''Load model''')

        # Page footer
        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def reset_widgets(self):
        """
        Reset check bar values
        :return: empty values in the widgets
        """

        pass

    def load_flow(self):
        """
        Move to load model flow
        :return: load model window
        """

        self.controller.set_new_model_running(False)
        self.controller.show_frame("LoadModel")

    def new_flow(self):
        """
        Move to new model flow
        :return: new model window
        """

        self.controller.set_new_model_running(True)
        self.controller.show_frame("NewModel")
