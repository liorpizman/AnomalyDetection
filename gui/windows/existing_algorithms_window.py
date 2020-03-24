#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import os
import win32api

from tkinter import END
from tkinter.ttk import Combobox
from gui.widgets.menubar import Menubar
from gui.shared.helper_methods import CROSS_WINDOWS_SETTINGS, set_path
from gui.shared.inputs_validation_helper import is_valid_model_paths
from gui.widgets_configurations.helper_methods import set_widget_to_left, set_logo_configuration, \
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


class ExistingAlgorithmsWindow(tk.Frame):

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
            text='''Please insert paths of existing models_1.''')
        set_widget_to_left(self.instructions)

        self.algorithms = dict()
        self.browse_buttons = dict()
        self.input_entries = dict()

        # LSTM existing algorithm
        self.lstm_var = tk.IntVar()
        self.lstm_check_button = tk.Checkbutton(self)
        self.lstm_check_button.place(relx=0.015, rely=0.38, height=32, width=146)
        self.lstm_check_button.configure(text="LSTM",
                                         variable=self.lstm_var,
                                         command=lambda: self.set_input_entry("LSTM", self.lstm_var.get()))
        set_widget_to_left(self.lstm_check_button)

        self.lstm_input = tk.Entry(self)
        self.lstm_input.place(relx=0.195, rely=0.38, height=25, relwidth=0.624)
        self.lstm_input.configure(state='disabled')

        self.lstm_btn = tk.Button(self, command=lambda: self.set_algorithm_path("LSTM"))
        self.lstm_btn.place(relx=0.833, rely=0.38, height=25, width=60)
        self.lstm_btn.configure(state='disabled')
        set_button_configuration(self.lstm_btn, text='''Browse''')

        self.browse_buttons["LSTM"] = self.lstm_btn
        self.input_entries["LSTM"] = self.lstm_input

        # One Class SVM existing algorithm
        self.ocsvm_var = tk.IntVar()
        self.ocsvm_check_button = tk.Checkbutton(self)
        self.ocsvm_check_button.place(relx=0.015, rely=0.47, height=32, width=146)
        self.ocsvm_check_button.configure(text="One Class SVM",
                                          variable=self.ocsvm_var,
                                          command=lambda: self.set_input_entry("OCSVM", self.lstm_var.get()))
        set_widget_to_left(self.ocsvm_check_button)

        self.ocsvm_input = tk.Entry(self)
        self.ocsvm_input.place(relx=0.195, rely=0.47, height=25, relwidth=0.624)
        self.ocsvm_input.configure(state='disabled')

        self.ocsvm_btn = tk.Button(self, command=lambda: self.set_algorithm_path("OCSVM"))
        self.ocsvm_btn.place(relx=0.833, rely=0.47, height=25, width=60)
        self.ocsvm_btn.configure(state='disabled')
        set_button_configuration(self.ocsvm_btn, text='''Browse''')

        self.browse_buttons["OCSVM"] = self.ocsvm_btn
        self.input_entries["OCSVM"] = self.ocsvm_input

        # KNN existing algorithm
        self.knn_var = tk.IntVar()
        self.knn_check_button = tk.Checkbutton(self)
        self.knn_check_button.place(relx=0.015, rely=0.56, height=32, width=146)
        self.knn_check_button.configure(text="KNN",
                                        variable=self.knn_var,
                                        command=lambda: self.set_input_entry("KNN", self.knn_var.get()))
        set_widget_to_left(self.knn_check_button)

        self.knn_input = tk.Entry(self)
        self.knn_input.place(relx=0.195, rely=0.56, height=25, relwidth=0.624)
        self.knn_input.configure(state='disabled')

        self.knn_btn = tk.Button(self, command=lambda: self.set_algorithm_path("KNN"))
        self.knn_btn.place(relx=0.833, rely=0.56, height=25, width=60)
        self.knn_btn.configure(state='disabled')
        set_button_configuration(self.knn_btn, text='''Browse''')

        self.browse_buttons["KNN"] = self.knn_btn
        self.input_entries["KNN"] = self.knn_input

        # Isolation Forest existing algorithm
        self.isolation_forest_var = tk.IntVar()
        self.isolation_forest_check_button = tk.Checkbutton(self)
        self.isolation_forest_check_button.place(relx=0.015, rely=0.65, height=32, width=146)
        self.isolation_forest_check_button.configure(text="Isolation forest",
                                                     variable=self.isolation_forest_var,
                                                     command=lambda: self.set_input_entry("isolation_forest",
                                                                                          self.isolation_forest_var.get()))
        set_widget_to_left(self.isolation_forest_check_button)

        self.isolation_forest_input = tk.Entry(self)
        self.isolation_forest_input.place(relx=0.195, rely=0.65, height=25, relwidth=0.624)
        self.isolation_forest_input.configure(state='disabled')

        self.isolation_forest_btn = tk.Button(self, command=lambda: self.set_algorithm_path("isolation_forest"))
        self.isolation_forest_btn.place(relx=0.833, rely=0.65, height=25, width=60)
        self.isolation_forest_btn.configure(state='disabled')
        set_button_configuration(self.isolation_forest_btn, text='''Browse''')

        self.browse_buttons["isolation_forest"] = self.isolation_forest_btn
        self.input_entries["isolation_forest"] = self.isolation_forest_input

        # Threshold
        self.threshold_list = [0.9, 0.8, 0.7]

        self.threshold = tk.Label(self)
        self.threshold.place(relx=0.015, rely=0.74, height=32, width=150)
        self.threshold.configure(text='''Threshold''')
        set_widget_to_left(self.threshold)

        self.threshold_combo = Combobox(self, values=self.threshold_list)
        self.threshold_combo.place(relx=0.195, rely=0.74, height=25, relwidth=0.154)
        self.threshold_combo.current(0)

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

    def back_window(self):
        self.controller.set_new_model_running(False)
        self.controller.show_frame("MainWindow")

    def set_input_entry(self, entry_name, state):
        if state:
            self.browse_buttons[entry_name]['state'] = 'active'
            self.input_entries[entry_name]['state'] = 'normal'
            self.algorithms[entry_name] = ""
        else:
            self.input_entries[entry_name].delete(0, END)
            self.browse_buttons[entry_name]['state'] = 'disabled'
            self.input_entries[entry_name]['state'] = 'disabled'
            self.algorithms.pop(entry_name, None)

    def set_algorithm_path(self, algorithm):
        self.input_entries[algorithm].delete(0, END)
        path = set_path()
        self.input_entries[algorithm].insert(0, path)

    def next_window(self):
        self.update_selected_algorithms()
        if self.validate_next_step():
            self.set_load_model_parameters()

    def validate_next_step(self):
        if not self.algorithms:
            win32api.MessageBox(0, 'Please select algorithm & path for the model before the next step.',
                                'Invalid algorithm', 0x00001000)
            return False

        if not is_valid_model_paths(self.algorithms.values()):
            win32api.MessageBox(0, 'At least one of your inputs is invalid or not in type of .h5 file!',
                                'Invalid inputs', 0x00001000)
            return False
        return True

    def update_selected_algorithms(self):
        tmp_algorithms = dict()
        for algorithm in self.algorithms:
            tmp_algorithms[algorithm] = self.input_entries[algorithm].get()
        self.algorithms = tmp_algorithms

    def set_load_model_parameters(self):
        self.controller.set_existing_algorithms(self.algorithms)
        self.controller.set_existing_algorithms_threshold(float(self.threshold_combo.get()))
        self.controller.reinitialize_frame("SimilarityFunctionsWindow")
