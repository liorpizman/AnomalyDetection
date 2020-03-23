#! /usr/bin/env python
#  -*- coding: utf-8 -*-

import os
import threading

from tkinter.font import Font
from gui.animated_gif import AnimatedGif
from datetime import timedelta
from timeit import default_timer as timer
from string import Template
from gui.menubar import Menubar
from gui.utils.constants import LOADING_WINDOW_SETTINGS, CROSS_WINDOWS_SETTINGS
from gui.widgets_configurations.helper_methods import set_logo_configuration, set_copyright_configuration, \
    set_widget_to_left, set_button_configuration

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


class LoadingWindow(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.menubar = Menubar(controller)
        self.controller.option_add('*tearOff', 'FALSE')  # Disables ability to tear menu bar into own window
        system_logo = CROSS_WINDOWS_SETTINGS.get('LOGO')
        photo_location = os.path.join(system_logo)
        global logo_img
        logo_img = tk.PhotoImage(file=photo_location)

        # Page header
        self.logo_png = tk.Button(self)
        self.logo_png.place(relx=0.28, rely=0.029, height=172, width=300)
        set_logo_configuration(self.logo_png, image=logo_img)

        self.instructions = tk.Label(self)
        self.instructions.place(relx=0.015, rely=0.3, height=32, width=635)
        self.instructions.configure(text='''Loading model, please wait...''')
        set_widget_to_left(self.instructions)

        loading_gif = LOADING_WINDOW_SETTINGS.get('LOADING_GIF')
        delay_between_frames = LOADING_WINDOW_SETTINGS.get('DELAY_BETWEEN_FRAMES')

        self.title_font = Font(family='Helvetica', size=12, weight="bold")

        self.loading_gif = AnimatedGif(self, loading_gif, delay_between_frames)
        self.loading_gif.place(relx=0.1, rely=0.35, height=330, width=600)

        self.clock_label = tk.Label(self, text="", font=self.title_font)
        self.clock_label.place(relx=0.38, rely=0.7, height=32, width=150)

        # Page footer
        self.stop_button = tk.Button(self, command=self.stop_model_process)
        self.stop_button.place(relx=0.813, rely=0.839, height=25, width=81)
        set_button_configuration(self.stop_button, text='''Stop''')

        self.back_button = tk.Button(self, command=self.back_window)
        self.back_button.place(relx=0.017, rely=0.839, height=25, width=81)
        set_button_configuration(self.back_button, text='''Back''')
        self.back_button.configure(state='disabled')

        self.copyright = tk.Label(self)
        self.copyright.place(relx=0, rely=0.958, height=25, width=750)
        set_copyright_configuration(self.copyright)

        # Page logic
        self.loading_gif.start()

    def back_window(self):
        self.controller.show_frame("SimilarityFunctionsWindow")

    def stop_model_process(self):
        self.back_button.configure(state='active')
        self.stop_button.configure(state='disabled')
        try:
            self.model_process_thread.join()
        except:
            pass

    def reinitialize(self):
        self.model_process_thread = threading.Thread(name='model_process', target=self.loading_process)
        self.model_process_thread.start()
        self.start_time = timer()
        self.update_clock()

    def update_clock(self):
        now = timer()
        duration = timedelta(seconds=now - self.start_time)
        self.clock_label.configure(text=strfdelta(duration, '%H:%M:%S'))
        self.controller.after(200, self.update_clock)

    def loading_process(self):
        self.controller.run_models()
        self.controller.show_frame("ResultsWindow")


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
