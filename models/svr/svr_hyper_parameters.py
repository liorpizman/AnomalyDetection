class svr_hyper_parameters:
    KERNEL = None
    GAMMA = None
    EPSILON = None
    THRESHOLD = None
    DEFAULT_EPSILON = 0.1
    DEFAULT_THRESHOLD = 0.99

    @staticmethod
    def set_kernel(kernel):
        svr_hyper_parameters.KERNEL = kernel

    @staticmethod
    def remove_kernel():
        svr_hyper_parameters.KERNEL = None

    @staticmethod
    def get_kernel():
        return svr_hyper_parameters.KERNEL

    @staticmethod
    def set_gamma(gamma):
        svr_hyper_parameters.GAMMA = gamma

    @staticmethod
    def remove_gamma():
        svr_hyper_parameters.GAMMA = None

    @staticmethod
    def get_gamma():
        return svr_hyper_parameters.GAMMA

    @staticmethod
    def set_epsilon(epsilon):
        try:
            svr_hyper_parameters.EPSILON = float(epsilon)
        except:
            svr_hyper_parameters.EPSILON = svr_hyper_parameters.DEFAULT_EPSILON

    @staticmethod
    def remove_epsilon():
        svr_hyper_parameters.EPSILON = None

    @staticmethod
    def get_epsilon():
        return svr_hyper_parameters.EPSILON

    @staticmethod
    def set_threshold(threshold):
        try:
            svr_hyper_parameters.THRESHOLD = float(threshold)
        except:
            svr_hyper_parameters.THRESHOLD = svr_hyper_parameters.DEFAULT_THRESHOLD

    @staticmethod
    def remove_threshold():
        svr_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return svr_hyper_parameters.THRESHOLD
