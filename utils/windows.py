'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants of attacks windows' sizes in Automatic dependent surveillance – broadcast (ADS-B) dataset
'''

# from utils.settings import LSTM_WINDOW_SIZE

windows = {
    "flight": [
        {
            "lower": 180,
            "upper": 250,
        }
    ],
    "flight_lstm": [
        {
            # "lower": 180 - LSTM_WINDOW_SIZE + 1,
            "upper": 249
        }
    ]
}
