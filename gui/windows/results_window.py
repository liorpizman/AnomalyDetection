#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from ipython_genutils.py3compat import xrange
from gui.widgets.menubar import Menubar
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets.table.table import Table
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

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(text='''Results''')
        set_widget_to_left(self.instructions)

        self.metrics_table = Table(self,
                                   columns=["Metric", "Down attack", "Up attack", "Fore attack", "Random attack"],
                                   column_minwidths=[None, None, None])
        self.metrics_table.pack(expand=True, fill=X, padx=10, pady=10)

        # Page footer
        self.back_button = tk.Button(self, command=lambda: controller.show_frame("MainWindow"))
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def reinitialize(self):
        try:
            results_data = InputSettings.get_results_metrics_data()
            chosen_algorithms = InputSettings.get_existing_algorithms()
            flight_routes = InputSettings.get_flight_routes()

            data = results_data["LSTM"]["mexico_las_veags"]  # should be changed to dynamic table

            attacks_columns = list(data.values())[0]

            # Creates a 2D array, all set to 0
            rows = len(data.keys())
            columns = len(attacks_columns)
            zero_matrix = [[0 for i in xrange(columns)] for i in xrange(rows)]
            self.metrics_table.set_data(zero_matrix)

            for i, metric in enumerate(data.keys()):
                attacks_data = data[metric]
                self.metrics_table.cell(i, 0, metric)
                for j, attack in enumerate(attacks_data.keys()):
                    self.metrics_table.cell(i, j + 1, attacks_data[attack])
        except Exception:
            pass
