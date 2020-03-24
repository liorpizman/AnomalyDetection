#! /usr/bin/env python
#  -*- coding: utf-8 -*-

import os

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
            text='''Please choose similarity function from the following options:''')
        set_widget_to_left(self.instructions)

        self.similarity_functions = Checkbar(self, load_similarity_list(), checkCallback=self.set_similarity_score)
        self.similarity_functions.place(relx=0.1, rely=0.35, height=400, width=700)

        self.save_model_var = tk.IntVar()
        self.save_model_check_button = tk.Checkbutton(self,
                                                      text="Save model",
                                                      variable=self.save_model_var,
                                                      command=self.set_saving_model)

        # Page footer
        self.next_button = tk.Button(self, command=self.next_window)
        self.next_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.next_button, text='''Next''')

        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

    def set_similarity_score(self):
        similarity_list = set()
        for check, var in zip(self.similarity_functions.get_checks(),
                              self.similarity_functions.get_vars()):
            if var.get():  # show algorithms' options
                similarity_list.add(check.cget("text"))
        self.controller.set_similarity_score(similarity_list)

    def next_window(self):
        self.controller.reinitialize_frame("LoadingWindow")

    def back_window(self):
        is_new_model_flow = self.controller.get_new_model_running()
        if is_new_model_flow:
            self.controller.show_frame("AlgorithmsWindow")
        else:
            self.controller.show_frame("ExistingAlgorithmsWindow")

    def reinitialize(self):
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
            self.save_model_check_button.destroy()

    def set_saving_model(self):
        self.controller.set_saving_model(self.save_model_var.get() == 1)
