#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Animated GIF which is presented in the application
'''

import tkinter as tk


class AnimatedGif(tk.Label):
    """
     A Class used to show an animated GIF

    Methods
    -------
    stop()
            Description | Stop GIF animation

    reset_frames_count()
            Description | Reset frames count for next presentation

    display_frame()
            Description | Display a frame of the GIF

    start()
            Description | Looping through the frames

    """

    def __init__(self, root, file_path, frames_delay=0.03):
        """
        Parameters
        ----------

        :param root: the root controller
        :param file_path: the path of the GIF file
        :param frames_delay: delay between frames
        """

        tk.Label.__init__(self, root)
        self.root = root
        self.frames_delay = frames_delay
        self.file_path = file_path
        self.stop = False
        self.count_frames = 0

    def stop(self):
        """
        Stop GIF animation
        :return: Stop GIF loop
        """
        self.stop = True

    def reset_frames_count(self):
        """
        Reset frames count for next presentation
        :return: count_frames is equal to zero
        """
        self.count_frames = 0

    def display_frame(self):
        """
        Display a frame of the GIF
        :return: Specific frame is shown
        """
        self.animation = tk.PhotoImage(file=self.file_path, format='gif -index {}'.format(self.count_frames))
        self.configure(image=self.animation)
        self.count_frames += 1

    def start(self):
        """
        Looping through the frames
        :return: GIF is on
        """

        try:
            self.display_frame()
        except tk.TclError:
            self.reset_frames_count()
        if not self.stop:
            self.root.after(int(self.frames_delay * 1000), self.start)
