#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from gui.results_permutation_frame.results_permutation_frame import ResultsPermutationFrame
from gui.shared.constants import CROSS_WINDOWS_SETTINGS
from gui.widgets.menubar import Menubar
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_button_configuration, \
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


class ResultsTableWindow(tk.Frame):

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

        # Dynamic results table
        self.frame_height = 300
        self.frame_width = 450

        self.results_table = ResultsPermutationFrame(self)
        self.results_table.place(relx=0,
                                 rely=0,
                                 height=self.frame_height,
                                 width=self.frame_width)

        # Page footer
        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def back_window(self):
        self.controller.reinitialize_frame("ResultsWindow")

    def reinitialize(self):
        self.reinitialize_results_table()

    def reinitialize_results_table(self):
        self.results_table.destroy()
        self.results_table = ResultsPermutationFrame(self)
        self.results_table.place(relx=0.1,
                                 rely=0.3,
                                 height=self.frame_height,
                                 width=self.frame_width)
