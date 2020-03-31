#! /usr/bin/env python
#  -*- coding: utf-8 -*-

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
    """Base class for cells"""
