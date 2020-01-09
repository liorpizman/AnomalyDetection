import tkinter as tk

from gui.lstm_frame_options import LSTMFrameOptions


class LSTMWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.title = tk.Label(self, text="LSTM Parameters", font=controller.title_font)
        self.options_to_show = LSTMFrameOptions(self)

        self.back_button = tk.Button(self, text="Cancel",
                                     command=lambda: self.controller.show_frame("AlgorithmsWindow"))

        self.save_button = tk.Button(self, text="Save parameters",
                                     command=lambda: self.save_algorithm_parameters(
                                         self.options_to_show.get_algorithm_parameters()))

        self.title.grid(row=0, column=2, pady=3)
        self.options_to_show.grid(row=2, column=2, pady=3)
        self.back_button.grid(row=48, column=2, pady=3)
        self.save_button.grid(row=48, column=15, pady=3)

    def set_algorithm_parameters(self, algorithm_parameters):
        self.controller.set_algorithm_parameters("LSTM", algorithm_parameters)

    def save_algorithm_parameters(self, algorithm_parameters):
        self.set_algorithm_parameters(algorithm_parameters)
        self.controller.show_frame("AlgorithmsWindow")
