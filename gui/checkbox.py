from tkinter import *


class Checkbar(Frame):

    def __init__(self, parent=None, picks=[], editButtons=False, checkCallback=None, buttonCallback=None):
        Frame.__init__(self, parent)
        self.vars = []
        self.checks = []
        self.buttons = []
        row = 2
        col = 2
        enable_functionality = 'active'
        for pick in picks:
            var = IntVar()
            if pick != "LSTM" and pick != "Cosine similarity":
                enable_functionality = 'disabled'

            check_button = Checkbutton(self, text=pick, variable=var, state=enable_functionality,
                                       command=lambda: self.set_button_state(checkCallback))
            check_button.grid(sticky="W", row=row, column=col)
            if editButtons:
                edit_button = Button(self, text=pick + " edit", state='disabled', command=buttonCallback)
                edit_button.grid(sticky="W", row=row, column=col + 2)
                self.buttons.append(edit_button)
            self.grid_rowconfigure(row, minsize=60)
            self.vars.append(var)
            self.checks.append(check_button)
            row = row + 50

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
