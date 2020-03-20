#! /usr/bin/env python
#  -*- coding: utf-8 -*-

from tkinter.ttk import Combobox
from gui.utils.helper_methods import load_lstm_activation_list, load_lstm_loss_list, load_lstm_optimizer_list, \
    load_lstm_window_size_list, load_lstm_encoder_dimension_list, load_lstm_threshold_list
from gui.widgets_configurations.helper_methods import set_widget_to_left

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


class LSTMFrameOptions(tk.Frame):

    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parameters = {}
        self.activation = load_lstm_activation_list()
        self.loss = load_lstm_loss_list()
        self.optimizer = load_lstm_optimizer_list()
        self.window_size = load_lstm_window_size_list()
        self.encoder_dimension = load_lstm_encoder_dimension_list()
        self.encoder_threshold = load_lstm_threshold_list()

        # LSTM activation parameter
        self.lstm_activation = tk.Label(self)
        self.lstm_activation.place(relx=0.015, rely=0, height=25, width=150)
        self.lstm_activation.configure(text='''LSTM activation''')
        set_widget_to_left(self.lstm_activation)

        self.lstm_activation_combo = Combobox(self, values=self.activation)
        self.lstm_activation_combo.place(relx=0.285, rely=0, height=25, width=150)
        self.lstm_activation_combo.current(0)
        self.parameters["activation"] = self.lstm_activation_combo

        # LSTM loss parameter
        self.lstm_loss = tk.Label(self)
        self.lstm_loss.place(relx=0.015, rely=0.1, height=25, width=150)
        self.lstm_loss.configure(text='''LSTM loss''')
        set_widget_to_left(self.lstm_loss)

        self.lstm_loss_combo = Combobox(self, values=self.loss)
        self.lstm_loss_combo.place(relx=0.285, rely=0.1, height=25, width=150)
        self.lstm_loss_combo.current(0)
        self.parameters["loss"] = self.lstm_loss_combo

        # LSTM optimizer parameter
        self.lstm_optimizer = tk.Label(self)
        self.lstm_optimizer.place(relx=0.015, rely=0.2, height=25, width=150)
        self.lstm_optimizer.configure(text='''LSTM optimizer''')
        set_widget_to_left(self.lstm_optimizer)

        self.lstm_optimizer_combo = Combobox(self, values=self.optimizer)
        self.lstm_optimizer_combo.place(relx=0.285, rely=0.2, height=25, width=150)
        self.lstm_optimizer_combo.current(0)
        self.parameters["optimizer"] = self.lstm_optimizer_combo

        # LSTM window size parameter
        self.lstm_window_size = tk.Label(self)
        self.lstm_window_size.place(relx=0.015, rely=0.3, height=25, width=150)
        self.lstm_window_size.configure(text='''LSTM window size''')
        set_widget_to_left(self.lstm_window_size)

        self.lstm_window_combo = Combobox(self, values=self.window_size)
        self.lstm_window_combo.place(relx=0.285, rely=0.3, height=25, width=150)
        self.lstm_window_combo.current(0)
        self.parameters["window_size"] = self.lstm_window_combo

        # LSTM encoding dimension parameter
        self.lstm_encoding_dimension = tk.Label(self)
        self.lstm_encoding_dimension.place(relx=0.015, rely=0.4, height=25, width=150)
        self.lstm_encoding_dimension.configure(text='''LSTM encoding dimension''')
        set_widget_to_left(self.lstm_encoding_dimension)

        self.lstm_encoder_dimension_combo = Combobox(self, values=self.encoder_dimension)
        self.lstm_encoder_dimension_combo.place(relx=0.285, rely=0.4, height=25, width=150)
        self.lstm_encoder_dimension_combo.current(0)
        self.parameters["encoding_dimension"] = self.lstm_encoder_dimension_combo

        # LSTM threshold parameter
        self.lstm_threshold = tk.Label(self)
        self.lstm_threshold.place(relx=0.015, rely=0.5, height=25, width=150)
        self.lstm_threshold.configure(text='''LSTM threshold''')
        set_widget_to_left(self.lstm_threshold)

        self.lstm_threshold_combo = Combobox(self, values=self.encoder_threshold)
        self.lstm_threshold_combo.place(relx=0.285, rely=0.5, height=25, width=150)
        self.lstm_threshold_combo.current(0)
        self.parameters["threshold"] = self.lstm_threshold_combo

    def get_algorithm_parameters(self):  # This function can be done in a single line - check how to do it right
        sol = {}
        for parameter in self.parameters.keys():
            sol[parameter] = self.parameters[parameter].get()
        return sol
