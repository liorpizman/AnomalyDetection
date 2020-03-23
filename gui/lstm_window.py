#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from gui.lstm_frame_options import LSTMFrameOptions
from gui.menubar import Menubar
from gui.utils.constants import CROSS_WINDOWS_SETTINGS
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


class LSTMWindow(tk.Frame):

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
            text='''Please select the values for each of the following parameters:''')
        set_widget_to_left(self.instructions)

        self.options_to_show = LSTMFrameOptions(self)
        self.options_to_show.place(relx=0.1, rely=0.35, height=400, width=650)

        # Page footer
        self.next_button = tk.Button(self, command=lambda: self.save_algorithm_parameters(
            self.options_to_show.get_algorithm_parameters()))
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Save''')

        self.back_button = tk.Button(self, command=lambda: self.controller.show_frame("AlgorithmsWindow"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Cancel''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def set_algorithm_parameters(self, algorithm_parameters):
        self.controller.set_algorithm_parameters("LSTM", algorithm_parameters)

    def save_algorithm_parameters(self, algorithm_parameters):
        self.set_algorithm_parameters(algorithm_parameters)
        self.controller.show_frame("AlgorithmsWindow")
