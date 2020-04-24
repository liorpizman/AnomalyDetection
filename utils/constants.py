'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants used to handle repeatable values in model controller
'''

ATTACK_COLUMN = 'GPS Spoofing'

COLUMNS_TO_REMOVE = [
    'label',
    'flight_id',
    'Home IP',
    'RSSI0 PARABOLIC',
    'RSSI1 PARABOLIC',
    'CINR0 PARABOLIC',
    'CINR1 PARABOLIC',
    'Modem Lock',
    'Drone IP',
    'Home to drone bearing',
    'Home to drone elevation',
    'Real Time',
    'GPS Spoofing'
]

NON_ATTACK_VALUE = 0

ATTACK_VALUE = 1
