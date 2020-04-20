'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants used to handle repeatable values in model controller
'''

ATTACK_COLUMN = 'GPS Spoofing'
COLUMNS_TO_REMOVE = ['label', 'flight_id', 'Route Index', 'Route Time', 'Home IP', 'Modem Lock', 'Drone IP',
                     'Real Time']
NON_ATTACK_VALUE = 0
ATTACK_VALUE = 1
