import tkinter as tk
from tkinter import YES, BOTH, NW


class LoadingWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Create Widgets
        self.laoding_title = tk.Label(self, text="Loading Model Creation...", font=controller.title_font)

        # Layout using grid
        self.laoding_title.grid(row=0, column=2, pady=3)
