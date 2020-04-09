#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Table cell which is presented in the application
'''

try:
    from Tkinter import Frame, Label, Message, StringVar
    from Tkconstants import *
except ImportError:
    from tkinter import Frame, Label, Message, StringVar
    from tkinter.constants import *

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


class Cell(Frame):
    """
    A Class used as a base class for table cells
    """
