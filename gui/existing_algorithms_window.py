import tkinter as tk
import win32api
from tkinter import END
from tkinter.ttk import Combobox

from gui.utils.inputs_validation_helper import is_valid_model_paths
from gui.utils.helper_methods import set_path, set_file_path
from utils.shared.helper_methods import is_valid_directory


class ExistingAlgorithmsWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.controller.geometry('800x600')
        # Create Widgets
        self.row = 0
        self.col = 0
        self.algorithms = dict()
        self.browse_buttons = dict()
        self.input_entries = dict()

        self.existing_algorithms_title = tk.Label(self, text="Existing algorithms", font=controller.title_font)

        self.lstm_var = tk.IntVar()
        self.lstm_check_button = tk.Checkbutton(self, text="LSTM", variable=self.lstm_var,
                                                command=lambda: self.set_input_entry("LSTM", self.lstm_var.get()))
        self.lstm_input = tk.Entry(self, state='disabled', width=80)
        self.lstm_btn = tk.Button(self, text="Browse", state='disabled',
                                  command=lambda: self.set_algorithm_path("LSTM"))
        self.browse_buttons["LSTM"] = self.lstm_btn
        self.input_entries["LSTM"] = self.lstm_input

        self.row += 1
        self.col += 1

        self.ocsvm_var = tk.IntVar()
        self.ocsvm_check_button = tk.Checkbutton(self, state='disabled', text="One Class SVM", variable=self.ocsvm_var,
                                                 command=lambda: self.set_input_entry("OCSVM", self.lstm_var.get()))
        self.ocsvm_input = tk.Entry(self, state='disabled', width=80)
        self.ocsvm_btn = tk.Button(self, text="Browse", state='disabled',
                                   command=lambda: self.set_algorithm_path("OCSVM"))
        self.browse_buttons["OCSVM"] = self.ocsvm_btn
        self.input_entries["OCSVM"] = self.ocsvm_input

        self.row += 1
        self.col += 1

        self.htm_var = tk.IntVar()
        self.htm_check_button = tk.Checkbutton(self, state='disabled', text="HTM", variable=self.htm_var,
                                               command=lambda: self.set_input_entry("HTM", self.lstm_var.get()))
        self.htm_input = tk.Entry(self, state='disabled', width=80)
        self.htm_btn = tk.Button(self, text="Browse", state='disabled', command=lambda: self.set_algorithm_path("HTM"))
        self.browse_buttons["HTM"] = self.htm_btn
        self.input_entries["HTM"] = self.htm_input

        self.row += 1
        self.col += 1

        self.isolation_forest_var = tk.IntVar()
        self.isolation_forest_check_button = tk.Checkbutton(self, state='disabled', text="isolation_forest",
                                                            variable=self.isolation_forest_var,
                                                            command=lambda: self.set_input_entry("isolation_forest",
                                                                                                 self.lstm_var.get()))
        self.isolation_forest_input = tk.Entry(self, state='disabled', width=80)
        self.isolation_forest_btn = tk.Button(self, text="Browse", state='disabled',
                                              command=lambda: self.set_algorithm_path("isolation_forest"))
        self.browse_buttons["isolation_forest"] = self.isolation_forest_btn
        self.input_entries["isolation_forest"] = self.isolation_forest_input

        self.threshold_list = [0.9, 0.8, 0.7]
        tk.Label(self, text="Threshold").grid(sticky="W", row=6, column=0)
        self.threshold_combo = Combobox(self, state="readonly", values=self.threshold_list)
        self.threshold_combo.grid(sticky="W", row=6, column=1, columnspan=2)
        self.threshold_combo.current(0)

        self.back_button = tk.Button(self, text="Back",
                                     command=self.back_window)
        self.next_button = tk.Button(self, text="Next",
                                     command=self.next_window)
        # Layout using grid
        self.existing_algorithms_title.grid(row=0, column=1, pady=3)

        self.lstm_check_button.grid(sticky="W", row=1, column=0, pady=3)
        self.lstm_input.grid(row=1, column=1, pady=3, padx=10)
        self.lstm_btn.grid(row=1, column=2, pady=3)

        self.ocsvm_check_button.grid(sticky="W", row=2, column=0, pady=3)
        self.ocsvm_input.grid(row=2, column=1, pady=3, padx=10)
        self.ocsvm_btn.grid(row=2, column=2, pady=3)

        self.htm_check_button.grid(sticky="W", row=3, column=0, pady=3)
        self.htm_input.grid(row=3, column=1, pady=3, padx=10)
        self.htm_btn.grid(row=3, column=2, pady=3)

        self.isolation_forest_check_button.grid(sticky="W", row=4, column=0, pady=3)
        self.isolation_forest_input.grid(row=4, column=1, pady=3, padx=10)
        self.isolation_forest_btn.grid(row=4, column=2, pady=3)

        self.back_button.grid(row=15, column=0, pady=3)
        self.next_button.grid(row=15, column=3, pady=3)

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
        path = set_file_path()
        self.input_entries[algorithm].insert(0, path)

    def next_window(self):
        self.update_selected_algorithms()
        if self.validate_next_step():
            self.set_load_model_parameters()

    def get_features_columns_options(self):
        return self.controller.get_features_columns_options()

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
        self.controller.show_frame("FeatureSelectionWindow")
