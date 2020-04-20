#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Table data cell which is presented in the application
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


class Data_Cell(Cell):
    """
     A Class used as a data cell for a table widget

    Methods
    -------
    _on_configure()
            Description | Message widget configuration

    """

    def __init__(self, master, variable, anchor=CENTER, bordercolor=None, borderwidth=1, padx=0, pady=0,
                 background=None,
                 foreground=None, font=None):
        """
        Parameters
        ----------

        :param master: the master of the data cell
        :param variable: data variable
        :param anchor: anchor
        :param bordercolor: the color of the border
        :param borderwidth: the width of the data cell border
        :param padx: x coordinate padding
        :param pady: y coordinate padding
        :param background: background color
        :param foreground: foreground color
        :param font: text font
        """
        Cell.__init__(self,
                      master,
                      background=background,
                      highlightbackground=bordercolor,
                      highlightcolor=bordercolor,
                      highlightthickness=borderwidth,
                      bd=0)

        self._message_widget = Message(self,
                                       textvariable=variable,
                                       font=font,
                                       background=background,
                                       foreground=foreground)

        self._message_widget.pack(expand=True,
                                  padx=padx,
                                  pady=pady,
                                  anchor=anchor)

        self._message_widget.configure(width=100, pady=5)

    # Note: This block of code causing flickering in results table window
    # Solution source: https://stackoverflow.com/questions/17747904/continuous-call-of-the-configure-event-in-tkinter

    #     self.bind("<Configure>", self._on_configure)
    #
    # def _on_configure(self, event):
    #     """
    #     Message widget configuration
    #     :param event: event
    #     :return: configures message widget
    #     """
    #
    #     self._message_widget.configure(width=event.width)
