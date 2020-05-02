'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Constants used to handle repeatable values in gui package
'''

ANOMALY_DETECTION_METHODS = 'anomaly_detection_methods'
FEATURE_SELECTION_METHODS = 'feature_selection_methods'
SIMILARITY_FUNCTIONS = 'similarity_functions'

LSTM_ACTIVATION = 'lstm_activation'
LSTM_LOSS = 'lstm_loss'
LSTM_OPTIMIZER = 'lstm_optimizer'
LSTM_WINDOW = 'lstm_window'
LSTM_ENCODER_DIMENSION = 'lstm_encoder_dimension'
LSTM_THRESHOLD_FROM_TRAINING_PERCENT = "lstm_threshold_from_training_percent"

LOADING_WINDOW_SETTINGS = {
    'LOADING_GIF': './images/loading.gif',
    'DELAY_BETWEEN_FRAMES': 0.02
}

CROSS_WINDOWS_SETTINGS = {
    'LOGO': './images/anomaly_detection_logo.png',
    'FEATURE_SELECTION': './images/feature_selection.png',
    'PARAMETERS_OPTIONS': './images/parameters_options.png',
    'INFORMATION_DIR': 'images',
    'INFORMATION_FILE': 'info.png',
    'BGU': './images/BGU.png',
    'ISRAEL_INNOVATION_AUTHORITY': './images/Israel_Innovation_Authority.png',
    'MINISTRY_OF_DEFENSE': './images/Ministry_of_Defense.png',
    'MOBILICOM': './images/Mobilicom.png',
    'RESULTS': './images/results.png'
}
