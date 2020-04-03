from tkinter import *

from gui.shared.helper_methods import load_anomaly_detection_list
from gui.widgets_configurations.helper_methods import set_widget_to_left


class Checkbar(Frame):

    def __init__(self, parent=None, picks=[], editButtons=False, checkCallback=None):
        Frame.__init__(self, parent)
        self.parent = parent
        self.vars = []
        self.checks = []
        self.buttons = []
        relY = 0
        relX = 0
        enable_functionality = 'active'
        for pick in picks:
            var = IntVar()

            algorithm_show_function = self.get_algorithm_show_function(str(pick))
            check_button = Checkbutton(self,
                                       text=pick,
                                       variable=var,
                                       state=enable_functionality,
                                       command=lambda: self.set_button_state(checkCallback))

            check_button.place(relx=relX, rely=relY, height=30, width=150)
            set_widget_to_left(check_button)

            if editButtons:
                edit_button = Button(self,
                                     text=pick + " configuration",
                                     state='disabled',
                                     command=algorithm_show_function)

                edit_button.place(relx=relX + 0.35, rely=relY, height=30, width=220)
                self.buttons.append(edit_button)
            self.vars.append(var)
            self.checks.append(check_button)
            relY = relY + 0.1

    def state(self):
        return map((lambda var: var.get()), self.vars)

    def get_checkbar_state(self):
        return self.checks, self.vars

    def get_checks(self):
        return self.checks

    def get_vars(self):
        return self.vars

    def set_button_state(self, checkCallback):
        for button, var in zip(self.buttons, self.vars):
            if var.get():
                button['state'] = 'active'
            else:
                button['state'] = 'disabled'
        if checkCallback is not None:
            checkCallback()

    def show_LSTM_options(self):
        self.parent.show_algorithms_options(load_anomaly_detection_list()[0])

    def show_SVR_options(self):
        self.parent.show_algorithms_options(load_anomaly_detection_list()[1])

    def show_KNN_options(self):
        self.parent.show_algorithms_options(load_anomaly_detection_list()[2])

    def show_Random_Forest_options(self):
        self.parent.show_algorithms_options(load_anomaly_detection_list()[3])

    def get_algorithm_show_function(self, algorithm_name):
        algorithms = load_anomaly_detection_list()
        switcher = {
            algorithms[0]: self.show_LSTM_options,
            algorithms[1]: self.show_SVR_options,
            algorithms[2]: self.show_KNN_options,
            algorithms[3]: self.show_Random_Forest_options
        }
        return switcher.get(algorithm_name, None)
