class ocsvm_hyper_parameters:
    KERNEL = None
    GAMMA = None
    SHRINKING = None
    THRESHOLD = None

    @staticmethod
    def set_kernel(kernel):
        ocsvm_hyper_parameters.KERNEL = kernel

    @staticmethod
    def remove_neighbors_number():
        ocsvm_hyper_parameters.KERNEL = None

    @staticmethod
    def get_neighbors_number():
        return ocsvm_hyper_parameters.KERNEL

    @staticmethod
    def set_gamma(gamma):
        ocsvm_hyper_parameters.GAMMA = gamma

    @staticmethod
    def remove_gamma():
        ocsvm_hyper_parameters.GAMMA = None

    @staticmethod
    def get_gamma():
        return ocsvm_hyper_parameters.GAMMA

    @staticmethod
    def set_shrinking(shrinking):
        ocsvm_hyper_parameters.SHRINKING = shrinking

    @staticmethod
    def remove_shrinking():
        ocsvm_hyper_parameters.SHRINKING = None

    @staticmethod
    def get_shrinking():
        return ocsvm_hyper_parameters.SHRINKING

    @staticmethod
    def set_threshold(threshold):
        ocsvm_hyper_parameters.THRESHOLD = float(threshold)

    @staticmethod
    def remove_threshold():
        ocsvm_hyper_parameters.THRESHOLD = None

    @staticmethod
    def get_threshold():
        return ocsvm_hyper_parameters.THRESHOLD
