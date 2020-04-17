'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Algorithms mapper from BE side to FE side
'''


def algorithms_mapper(algorithm):
    """
    Switch between back end name to front end name
    :param algorithm: BE name
    :return: FE name
    """

    switcher = {
        "lstm": "LSTM",
        "svr": "SVR",
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest"
    }

    return switcher.get(algorithm, None)
