class knn_hyper_parameters:
    NEIGHBORS_NUMBER = None
    WEIGHTS = None
    ALGORITHM = None
    THRESHOLD = None

    @staticmethod
    def set_n_neighbors(n_neighbors):
        knn_hyper_parameters.NEIGHBORS_NUMBER = int(n_neighbors)

    @staticmethod
    def remove_n_neighbors():
        knn_hyper_parameters.NEIGHBORS_NUMBER = None

    @staticmethod
    def get_n_neighbors():
        return knn_hyper_parameters.NEIGHBORS_NUMBER

    @staticmethod
    def set_weights(weights):
        knn_hyper_parameters.WEIGHTS = weights

    @staticmethod
    def remove_weights():
        knn_hyper_parameters.WEIGHTS = None

    @staticmethod
    def get_weights():
        return knn_hyper_parameters.WEIGHTS

    @staticmethod
    def set_algorithm(algorithm):
        knn_hyper_parameters.ALGORITHM = algorithm

    @staticmethod
    def remove_algorithm():
        knn_hyper_parameters.ALGORITHM = None

    @staticmethod
    def get_algorithm():
        return knn_hyper_parameters.ALGORITHM

    @staticmethod
    def set_threshold(threshold):
        knn_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        knn_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return knn_hyper_parameters.THRESHOLD
