#! /usr/bin/env python
#  -*- coding: utf-8 -*-

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
    def __init__(self, master, variable, anchor=W, bordercolor=None, borderwidth=1, padx=0, pady=0, background=None,
                 foreground=None, font=None):
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

        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        self._message_widget.configure(width=event.width)
