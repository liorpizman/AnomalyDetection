#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Menu bar messages which triggered from the menu bar of the application
'''

TITLE_PREFIX = 'Anomaly Detection System'

HELP = {
    'TITLE': TITLE_PREFIX + ' - Help',
    'MESSAGE': 'Please use system\'s readMe file.\n'
               'For any other question you can send us an email:\n'
               '[pezman@post.bgu.ac.il] or [yehudap@post.bgu.ac.il]'
}

ABOUT = {
    'TITLE': TITLE_PREFIX + ' - About',
    'MESSAGE': 'System\'s main goal is to create machine learning models for anomaly detection on UAVs. '
               'The system allows creation and loading of machine learning models by using dynamic inputs.'
}

ABOUT_US = {
    'TITLE': TITLE_PREFIX + ' - About Us',
    'MESSAGE': 'Lior Pizman & Yehuda Pashay - 4th year students, software and information systems engineering (SISE)'
               ' department (Ben-Gurion University of the Negev) BGU, Israel.'
}
