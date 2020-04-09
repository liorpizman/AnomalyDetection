#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Menu bar which is presented in the application
'''

import tkinter
from tkinter import ttk


class Menubar(ttk.Frame):
    """
     A Class used to build a menu bar for the top of the main window

    Methods
    -------
    on_exit()
            Description | Exit the program

    display_help()
            Description | Displays help document

    display_about()
            Description | Displays info about program

    init_menubar()
            Description | Init menu bar parameters

    """

    def __init__(self, parent, *args, **kwargs):
        """
        Init menubar for the top of parent window

        Parameters
        ----------

        parent: Window
            the controller of the menubar
        """
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_menubar()

    def on_exit(self):
        """
        Exit program
        :return: out of the system
        """
        quit()

    def display_help(self):
        """
        Displays help document
        :return: help info
        """
        pass

    def display_about(self):
        """
        Displays info about program
        :return: about info
        """
        pass

    def init_menubar(self):
        """
        Init menu bar parameters
        :return: menu bar shown in the screen
        """
        self.menubar = tkinter.Menu(self.root)

        # Creates a "File" menu
        self.menu_file = tkinter.Menu(self.menubar)

        # Adds an option to the menu
        self.menu_file.add_command(label='Exit', command=self.on_exit)

        # Adds File menu to the bar. Can also be used to create submenus
        self.menubar.add_cascade(menu=self.menu_file, label='File')

        # Creates a "Help" menu
        self.menu_help = tkinter.Menu(self.menubar)
        self.menu_help.add_command(label='Help', command=self.display_help)
        self.menu_help.add_command(label='About', command=self.display_about)
        self.menubar.add_cascade(menu=self.menu_help, label='Help')

        self.root.config(menu=self.menubar)
