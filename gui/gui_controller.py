import tkinter as tk
from tkinter import font as tkfont

from gui.algorithms_window import AlgorithmsWindow
from gui.feature_selection_window import FeatureSelectionWindow
from gui.existing_algorithms_window import ExistingAlgorithmsWindow
from gui.load_model_window import LoadModel
from gui.loading_window import LoadingWindow
from gui.lstm_window import LSTMWindow
from gui.main_window import MainWindow
from gui.model_controller import ModelController
from gui.new_model_window import NewModel
from gui.results_window import ResultsWindow
from gui.similarity_functions_window import SimilarityFunctionsWindow


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
        self.frames = {}
        for F in (MainWindow,
                  NewModel,
                  LoadModel,
                  AlgorithmsWindow,
                  FeatureSelectionWindow,
                  SimilarityFunctionsWindow,
                  ExistingAlgorithmsWindow,
                  LoadingWindow,
                  ResultsWindow,
                  LSTMWindow):
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


if __name__ == "__main__":
    app = AnomalyDetectionGUI()
    app.mainloop()
