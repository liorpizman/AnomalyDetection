'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants used to handle repeatable values in gui package
'''

import os
from pathlib import Path

ANOMALY_DETECTION_METHODS = 'anomaly_detection_methods'
FEATURE_SELECTION_METHODS = 'feature_selection_methods'
SIMILARITY_FUNCTIONS = 'similarity_functions'

LSTM_ACTIVATION = 'lstm_activation'
LSTM_LOSS = 'lstm_loss'
LSTM_OPTIMIZER = 'lstm_optimizer'
LSTM_WINDOW = 'lstm_window'
LSTM_ENCODER_DIMENSION = 'lstm_encoder_dimension'
LSTM_THRESHOLD_FROM_TRAINING_PERCENT = "lstm_threshold_from_training_percent"

ROOT_DIR = Path(__file__).parent.parent.parent

LOADING_WINDOW_SETTINGS = {
    'LOADING_GIF': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'loading.gif']),
    'STOP': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'stop.png']),
    'DELAY_BETWEEN_FRAMES': 0.02
}

CROSS_WINDOWS_SETTINGS = {
    'LOGO': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'anomaly_detection_logo.png']),
    'FEATURE_SELECTION': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'feature_selection.png']),
    'PARAMETERS_OPTIONS': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'parameters_options.png']),
    'INFORMATION_DIR': 'images',
    'INFORMATION_FILE': 'info.png',
    'BGU': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'BGU.png']),
    'ISRAEL_INNOVATION_AUTHORITY': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'Israel_Innovation_Authority.png']),
    'MINISTRY_OF_DEFENSE': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'Ministry_of_Defense.png']),
    'MOBILICOM': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'Mobilicom.png']),
    'RESULTS': os.path.join(*[str(ROOT_DIR), 'gui', 'images', 'results.png'])
}

ACTIVE_BACKGROUND = 'gray88'
