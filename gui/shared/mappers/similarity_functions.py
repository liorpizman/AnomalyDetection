'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Similarity functions mapper from BE side to FE side
'''


def similarity_functions_mapper(similarity_function):
    """
    Switch between back end name to front end name
    :param similarity_function: BE name
    :return: FE name
    """

    switcher = {
        "cosine_similarity": "Cosine similarity",
        "mahalanobis_distance": "Mahalanobis distance",
        "mse": "MSE",
        "euclidean_distance": "Euclidean distance"
    }

    return switcher.get(similarity_function, None)
