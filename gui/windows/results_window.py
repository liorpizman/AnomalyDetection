#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from ipython_genutils.py3compat import xrange
from gui.shared.helper_methods import set_widget_for_param, trim_unnecessary_chars, transform_list
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_widget_to_left, \
    set_copyright_configuration, set_button_configuration
from utils.input_settings import InputSettings

try:
    import Tkinter as tk
    from Tkconstants import *
except ImportError:
    import tkinter as tk
    from tkinter.constants import *

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


class ResultsWindow(tk.Frame):

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

        self.toggle_results_button = tk.Button(self, command=self.toggle_results)
        self.toggle_results_button.place(relx=0.8, rely=0.4, height=25, width=81)
        set_button_configuration(self.toggle_results_button, text='''Show results''')

        # Page footer
        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def back_window(self):
        self.controller.show_frame("MainWindow")

    def toggle_results(self):
        selected_algorithm = self.parameters['algorithm'].get()
        selected_flight_route = self.parameters['flight_route'].get()

        self.controller.set_results_selected_algorithm(selected_algorithm)
        self.controller.set_results_selected_flight_route(selected_flight_route)

        self.controller.reinitialize_frame("ResultsTableWindow")

    def reinitialize(self):
        chosen_algorithms = list(InputSettings.get_algorithms())
        flight_routes = list(InputSettings.get_flight_routes())

        transformed_chosen_algorithms = transform_list(chosen_algorithms)
        transformed_flight_routes = transform_list(flight_routes)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(text="Choose an algorithm and a flight route in order to get the results.")
        set_widget_to_left(self.instructions)

        # Algorithm and Flight route permutation choice
        self.parameters = {}

        set_widget_for_param(frame=self,
                             text="Algorithm:",
                             combobox_values=transformed_chosen_algorithms,
                             param_key="algorithm",
                             relative_x=0.05,
                             y_coordinate=0.4)

        set_widget_for_param(frame=self,
                             text="Flight route:",
                             combobox_values=transformed_flight_routes,
                             param_key="flight_route",
                             relative_x=0.4,
                             y_coordinate=0.4)
