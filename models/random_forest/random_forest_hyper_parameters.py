class random_forest_hyper_parameters:
    N_ESTIMATORS = None
    CRITERION = None
    MAX_FEATURES = None
    RANDOM_STATE = None
    THRESHOLD = None

    # N estimators parameter
    @staticmethod
    def set_n_estimators(n_estimators):
        random_forest_hyper_parameters.N_ESTIMATORS = int(n_estimators)

    @staticmethod
    def remove_n_estimators():
        random_forest_hyper_parameters.N_ESTIMATORS = None

    @staticmethod
    def get_n_estimators():
        return random_forest_hyper_parameters.N_ESTIMATORS

    # Criterion parameter
    @staticmethod
    def set_criterion(criterion):
        random_forest_hyper_parameters.CRITERION = criterion

    @staticmethod
    def remove_criterion():
        random_forest_hyper_parameters.CRITERION = None

    @staticmethod
    def get_criterion():
        return random_forest_hyper_parameters.CRITERION

    # Max features parameter
    @staticmethod
    def set_max_features(max_features):
        random_forest_hyper_parameters.MAX_FEATURES = max_features

    @staticmethod
    def remove_max_features():
        random_forest_hyper_parameters.MAX_FEATURES = None

    @staticmethod
    def get_max_features():
        return random_forest_hyper_parameters.MAX_FEATURES

    # Random state parameter
    @staticmethod
    def set_random_state(random_state):
        random_forest_hyper_parameters.RANDOM_STATE = float(random_state)

    @staticmethod
    def remove_random_state():
        random_forest_hyper_parameters.RANDOM_STATE = None

    @staticmethod
    def get_random_state():
        return random_forest_hyper_parameters.RANDOM_STATE

    # N estimators parameter
    @staticmethod
    def set_threshold(threshold):
        random_forest_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        random_forest_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return random_forest_hyper_parameters.THRESHOLD
