'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants of operating system routes used to handle repeatable values in model controller
'''

import os

DATA_DIR = "data"

EXPORT_DIR = "export"

DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")

FIG_DIR = os.path.join(EXPORT_DIR, "figures")

DATA_TRANSFORMED_DIR = os.path.join(DATA_DIR, "transformed")
