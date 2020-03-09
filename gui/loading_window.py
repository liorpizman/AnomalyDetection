import tkinter as tk

from tkinter import font as tkfont
from gui.animated_gif import AnimatedGif
from datetime import timedelta
from timeit import default_timer as timer
from string import Template

LOADING_GIF = 'loading.gif'
DELAY_BETWEEN_FRAMES = 0.02


class LoadingWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.title_font = tkfont.Font(family='Helvetica', size=12, weight="bold")

        # Create Widgets
        self.loading_title = tk.Label(self, text="Loading, please wait!", font=self.controller.title_font)
        self.loading_gif = AnimatedGif(self, LOADING_GIF, DELAY_BETWEEN_FRAMES)
        self.clock_label = tk.Label(self, text="", font=self.title_font)

        # Layout using grid
        self.loading_title.grid(row=0, column=2, pady=3)
        self.clock_label.grid(row=20, column=2, pady=3)
        self.loading_gif.grid(row=2, column=2, pady=3)
        self.loading_gif.start()

    def reinitialize(self):
        self.start_time = timer()
        self.update_clock()

    def update_clock(self):
        now = timer()
        duration = timedelta(seconds=now - self.start_time)
        self.clock_label.configure(text=strfdelta(duration, '%H:%M:%S'))
        self.controller.after(200, self.update_clock)


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)
