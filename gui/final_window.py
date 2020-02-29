import tkinter as tk
from tkinter import YES, BOTH, NW


class FinalWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Create Widgets
        self.loading_title = tk.Label(self, text="Model created Successfully!", font=controller.title_font)

        # Layout using grid
        self.loading_title.grid(row=0, column=2, pady=3)
