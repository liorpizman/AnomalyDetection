try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk
from tkinter import font as tkfont

from gui.windows.algorithms_window import AlgorithmsWindow
from gui.windows.existing_algorithms_window import ExistingAlgorithmsWindow
from gui.windows.load_model_window import LoadModel
from gui.windows.loading_window import LoadingWindow
from gui.windows.parameters_options_window import ParametersOptionsWindow
from gui.windows.main_window import MainWindow
from utils.model_controller import ModelController
from gui.windows.new_model_window import NewModel
from gui.windows.results_window import ResultsWindow
from gui.windows.similarity_functions_window import SimilarityFunctionsWindow


class AnomalyDetectionGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.model_controller = ModelController(self)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.geometry('800x600')
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # container.option_add('*tearOff', 'FALSE')  # Disables ability to tear menu bar into own window
        self.current_algorithm_to_edit = "LSTM"

        self.frames = {}
        for F in (MainWindow,
                  NewModel,
                  LoadModel,
                  AlgorithmsWindow,
                  SimilarityFunctionsWindow,
                  ExistingAlgorithmsWindow,
                  LoadingWindow,
                  ResultsWindow,
                  ParametersOptionsWindow):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainWindow")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def reinitialize_frame(self, page_name):
        '''Reinitialize_frame a frame for the given page name'''
        frame = self.frames[page_name]
        frame.reinitialize()
        frame.tkraise()

    def set_new_model_training_input_path(self, input):
        self.model_controller.set_training_data_path(input)

    def set_new_model_test_input_path(self, input):
        self.model_controller.set_test_data_path(input)

    def set_new_model_results_input_path(self, input):
        self.model_controller.set_results_path(input)

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.model_controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.model_controller.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def set_similarity_score(self, similarity_list):
        self.model_controller.set_similarity_score(similarity_list)

    def set_saving_model(self, save_model):
        self.model_controller.set_saving_model(save_model)

    def run_models(self):
        self.model_controller.run_models()

    def set_new_model_running(self, new_model_running):
        self.model_controller.set_new_model_running(new_model_running)

    def set_existing_algorithms(self, algorithms_dict):
        self.model_controller.set_existing_algorithms(algorithms_dict)

    def set_existing_algorithms_threshold(self, threshold):
        self.model_controller.set_existing_algorithms_threshold(threshold)

    def set_features_columns_options(self):
        self.model_controller.set_features_columns_options()

    def get_features_columns_options(self):
        return self.model_controller.get_features_columns_options()

    def set_users_selected_features(self, algorithm_name, features_list):
        self.model_controller.set_users_selected_features(algorithm_name, features_list)

    def add_new_thread(self, new_thread):
        self.model_controller.add_new_thread(new_thread)

    def get_existing_thread(self):
        return self.model_controller.get_existing_thread()

    def get_new_model_running(self):
        return self.model_controller.get_new_model_running()

    def set_current_algorithm_to_edit(self, algorithm_name):
        self.current_algorithm_to_edit = algorithm_name

    def get_current_algorithm_to_edit(self):
        return self.current_algorithm_to_edit

    def get_algorithms(self):
        return self.model_controller.get_algorithms()

    def remove_algorithm(self, algorithm_name):
        self.model_controller.remove_algorithm(algorithm_name)

    def remove_algorithms(self, algorithm_name):
        self.model_controller.remove_algorithms(algorithm_name)


if __name__ == "__main__":
    app = AnomalyDetectionGUI()
    app.mainloop()
