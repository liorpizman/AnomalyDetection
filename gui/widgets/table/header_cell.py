#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Table header cell which is presented in the application
'''

from gui.widgets.table.cell import Cell

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


class Header_Cell(Cell):
    """
     A Class used as a header cell for a table widget

    """

    def __init__(self, master, text, bordercolor=None, borderwidth=1, padx=None, pady=None, background=None,
                 foreground=None, font=None, anchor=CENTER):
        """
        Parameters
        ----------

        :param master: the master of the header cell
        :param text: the text to present in the header cell
        :param bordercolor: the color of the border
        :param borderwidth: the width of the border
        :param padx: x coordinate padding
        :param pady: y coordinate padding
        :param background: background color
        :param foreground: foreground color
        :param font: text font
        :param anchor: anchor
        """
        Cell.__init__(self,
                      master,
                      background=background,
                      highlightbackground=bordercolor,
                      highlightcolor=bordercolor,
                      highlightthickness=borderwidth,
                      bd=0)

        self._header_label = Label(self,
                                   text=text,
                                   background=background,
                                   foreground=foreground,
                                   font=font)
        self._header_label.pack(padx=padx,
                                pady=pady,
                                expand=True)

        if bordercolor is not None:
            separator = Frame(self,
                              height=2,
                              background=bordercolor,
                              bd=0,
                              highlightthickness=0,
                              class_="Separator")

            separator.pack(fill=X, anchor=anchor)
