#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os

from gui.menubar import Menubar
from gui.utils.constants import CROSS_WINDOWS_SETTINGS
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

    def __init__(self, parent, controller):
        '''This class configures and populates the main window.'''
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.menubar = Menubar(controller)
        self.controller.option_add('*tearOff', 'FALSE')  # Disables ability to tear menu bar into own window
        system_logo = CROSS_WINDOWS_SETTINGS.get('LOGO')
        photo_location = os.path.join(system_logo)
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        self.controller.geometry("700x550")
        self.controller.minsize(700, 550)
        self.controller.maxsize(700, 550)
        self.controller.resizable(1, 1)
        self.controller.title("Anomaly Detection Classifier")
        self.controller.configure(background="#eeeeee")

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        self.create_model_btn = tk.Button(self, command=lambda: self.controller.show_frame("NewModel"))
        self.create_model_btn.place(relx=0.41, rely=0.5, height=42, width=120)
        set_button_configuration(self.create_model_btn, text='''Create model''')

        self.load_model_btn = tk.Button(self, command=lambda: self.controller.show_frame("LoadModel"))
        self.load_model_btn.place(relx=0.41, rely=0.7, height=42, width=120)
        set_button_configuration(self.load_model_btn, text='''Load model''')

        # Page footer
        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)
