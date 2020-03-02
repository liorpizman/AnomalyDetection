import tkinter as tk

from gui.checkbox import Checkbar
from gui.utils.helper_methods import load_feature_selection_list
from utils.shared.input_settings import input_settings


class FeatureSelectionWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Create Widgets
        self.feature_selection_title = tk.Label(self, text="Choose feature selection methods",
                                                font=controller.title_font)

        self.feature_selection_methods = Checkbar(self, load_feature_selection_list())

        self.back_button = tk.Button(self, text="Back", command=self.back_window)

        self.next_button = tk.Button(self, text="Next", command=self.next_window)

        # Layout using grid
        self.feature_selection_title.grid(row=0, column=2, pady=3)
        self.feature_selection_methods.grid(row=2, column=2, pady=3)

        self.grid_rowconfigure(13, minsize=100)
        self.back_button.grid(row=50, column=2, pady=3)
        self.next_button.grid(row=50, column=15, pady=3)

    def back_window(self):
        new_model_running = input_settings.get_new_model_running()
        if new_model_running:
            self.controller.show_frame("AlgorithmsWindow")
        else:
            self.controller.show_frame("ExistingAlgorithmsWindow")

    def next_window(self):
        self.controller.show_frame("SimilarityFunctionsWindow")
