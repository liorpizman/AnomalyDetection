class isolation_forest_hyper_parameters:
    N_ESTIMATORS = None
    BOOTSTRAP = None
    MAX_FEATURES = None
    THRESHOLD = None

    @staticmethod
    def set_n_estimators(n_estimators):
        isolation_forest_hyper_parameters.N_ESTIMATORS = int(n_estimators)

    @staticmethod
    def remove_n_estimators():
        isolation_forest_hyper_parameters.N_ESTIMATORS = None

    @staticmethod
    def get_n_estimators():
        return isolation_forest_hyper_parameters.N_ESTIMATORS

    @staticmethod
    def set_bootstrap(bootstrap):
        isolation_forest_hyper_parameters.BOOTSTRAP = bootstrap

    @staticmethod
    def remove_bootstrap():
        isolation_forest_hyper_parameters.BOOTSTRAP = None

    @staticmethod
    def get_bootstrap():
        return isolation_forest_hyper_parameters.BOOTSTRAP

    @staticmethod
    def set_max_features(max_features):
        isolation_forest_hyper_parameters.MAX_FEATURES = max_features

    @staticmethod
    def remove_max_features():
        isolation_forest_hyper_parameters.MAX_FEATURES = None

    @staticmethod
    def get_max_features():
        return isolation_forest_hyper_parameters.MAX_FEATURES

    @staticmethod
    def set_threshold(threshold):
        isolation_forest_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        isolation_forest_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return isolation_forest_hyper_parameters.THRESHOLD
