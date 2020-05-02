#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Hover Button which is presented in the application
'''

import tkinter as tk

from gui.shared.constants import ACTIVE_BACKGROUND


class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        """
        Enter button coordinates
        :param e: event
        :return: new background
        """

        if self['state'] != 'disabled':
            self['background'] = ACTIVE_BACKGROUND

    def on_leave(self, e):
        """
        Leave button coordinates
        :param e: event
        :return: original background
        """

        self['background'] = self.defaultBackground
