#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Methods to handle repeatable actions which are done on widgets' configuration
'''

from gui.widgets_configurations.anchor_left import ANCHOR_LEFT
from gui.widgets_configurations.button import BUTTON_CONFIG
from gui.widgets_configurations.copyright import COPYRIGHT_CONFIG
from gui.widgets_configurations.logo import LOGO_CONFIG_INIT, LOGO_CONFIG_ADVANCED


def set_button_configuration(btn, text):
    """
    Set a configuration and a text for a given button
    :param btn: button widget
    :param text: the text on the button
    :return: configured button
    """

    btn.configure(BUTTON_CONFIG)
    btn.configure(text=text)


def set_logo_configuration(logo, image):
    """
    Set a configuration and an image for a given button
    :param logo: logo widget
    :param image: image to set on the logo widget
    :return: configured logo
    """

    logo.configure(LOGO_CONFIG_INIT)
    logo.configure(image=image)
    logo.configure(LOGO_CONFIG_ADVANCED)


def set_copyright_configuration(copy_right):
    """
    Set a configuration for a copy right label
    :param copy_right: input copy right label
    :return: configured copyright widget
    """

    copy_right.configure(COPYRIGHT_CONFIG)


def set_widget_to_left(widget):
    """
    Set a configuration for anchoring a widget to left side
    :param widget: input widget
    :return: copyright widget
    """

    widget.configure(ANCHOR_LEFT)
