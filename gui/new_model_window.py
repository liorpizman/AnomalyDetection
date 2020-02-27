import tkinter as tk
from tkinter import END

from gui.utils.helper_methods import set_training_path, set_test_path, set_path


class NewModel(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.geometry('800x600')
        # Create Widgets
        self.new_model_title = tk.Label(self, text="New model", font=controller.title_font)

        self.training_label = tk.Label(self, text="Training directory")
        self.training_input = tk.Entry(self, width=80)
        self.training_btn = tk.Button(self, text="Browse", command=self.set_input_path)

        self.test_label = tk.Label(self, text="Test directory")
        self.test_input = tk.Entry(self, width=80)
        self.test_btn = tk.Button(self, text="Browse", command=self.set_test_path)

        self.results_label = tk.Label(self, text="Results directory")
        self.results_input = tk.Entry(self, width=80)
        self.results_btn = tk.Button(self, text="Browse", command=self.set_results_path)

        self.back_button = tk.Button(self, text="Back",
                                     command=self.back_window)#lambda: controller.show_frame("MainWindow"))

        self.next_button = tk.Button(self, text="Next",
                                     command=self.next_window)#lambda: controller.show_frame("AlgorithmsWindow"))

        # Layout using grid
        self.new_model_title.grid(row=0, column=1, pady=3)

        self.training_label.grid(row=1, column=0, pady=3)
        self.training_input.grid(row=1, column=1, pady=3, padx=10)
        self.training_btn.grid(row=1, column=2, pady=3)

        self.test_input.grid(row=2, column=1, pady=3, padx=10)
        self.test_label.grid(row=2, column=0, pady=3)
        self.test_btn.grid(row=2, column=2, pady=3)

        self.results_input.grid(row=3, column=1, pady=3, padx=10)
        self.results_label.grid(row=3, column=0, pady=3)
        self.results_btn.grid(row=3, column=2, pady=3)

        self.back_button.grid(row=15, column=0, pady=3)
        self.next_button.grid(row=15, column=3, pady=3)

    def set_input_path(self):
        self.training_input.delete(0, END)
        path = set_path()
        self.training_input.insert(0, path)
        self.controller.set_new_model_training_input_path(path)

    def set_test_path(self):
        self.test_input.delete(0, END)
        path = set_path()
        self.test_input.insert(0, path)
        self.controller.set_new_model_test_input_path(path)

    def set_results_path(self):
        self.results_input.delete(0, END)
        path = set_path()
        self.results_input.insert(0, path)
        self.controller.set_new_model_results_input_path(path)

    def back_window(self):
        self.controller.set_new_model_running(False)
        self.controller.show_frame("MainWindow")

    def next_window(self):
        self.controller.set_new_model_running(True)
        self.controller.show_frame("AlgorithmsWindow")