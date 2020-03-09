import tkinter as tk
from gui.animated_gif import AnimatedGif

LOADING_GIF = 'loading.gif'
DELAY_BETWEEN_FRAMES = 0.02


class LoadingWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Create Widgets
        self.loading_title = tk.Label(self, text="Loading, please wait!", font=controller.title_font)
        self.loading_gif = AnimatedGif(self, LOADING_GIF, DELAY_BETWEEN_FRAMES)

        # Layout using grid
        self.loading_title.grid(row=0, column=2, pady=3)
        self.loading_gif.grid(row=2, column=2, pady=3)
        self.loading_gif.start()
