#! /usr/bin/env python
#  -*- coding: utf-8 -*-

'''
Anomaly Detection of GPS Spoofing Attacks on UAVs
Authors: Lior Pizman & Yehuda Pashay
GitHub: https://github.com/liorpizman/AnomalyDetection
DataSets: 1. ADS-B dataset 2. simulated data
---
Anomaly detection graphic user interface controller which is used to map all the client side actions
'''

from tkinter import font as tkfont
from gui.windows.algorithms_window import AlgorithmsWindow
from gui.windows.existing_algorithms_window import ExistingAlgorithmsWindow
from gui.windows.feature_selection_window import FeatureSelectionWindow
from gui.windows.loaded_data_window import LoadedDataWindow
from gui.windows.pre_tune_window import PreTuneModel
from gui.windows.results_plot_window import ResultsPlotWindow
from gui.windows.tune_model_window import TuneModel
from gui.windows.load_model_window import LoadModel
from gui.windows.loading_window import LoadingWindow
from gui.windows.parameters_options_window import ParametersOptionsWindow
from gui.windows.main_window import MainWindow
from gui.windows.results_table_window import ResultsTableWindow
from gui.windows.tune_results_window import TuneResultsWindow
from gui.windows.tuning_loading_window import TuningLoadingWindow
from utils.model_controller import ModelController
from gui.windows.new_model_window import NewModel
from gui.windows.results_window import ResultsWindow
from gui.windows.similarity_functions_window import SimilarityFunctionsWindow

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


class AnomalyDetectionGUI(tk.Tk):
    """
    A Class used to map all the actions in the GUI

    Methods
    -------

    init_main_controller()
            Description | Init main controller functionality

    show_frame(page_name)
            Description | Show the frame for a given page name

    reinitialize_frame(page_name)
            Description | Reinitialize a frame for a given page name

    reset_frame()
            Description | Reset all existing frames

    set_new_model_training_input_path(input)
            Description | Set the path of the train data set directory for new model flow

    set_new_model_test_input_path(input)
            Description | Set the path of the test data set directory for new model flow

    set_new_model_results_input_path(input)
            Description | Set the path of the results directory for new model flow

    set_algorithm_parameters(algorithm_name, algorithm_parameters)
            Description | Set the parameters which were chosen by the user to a given algorithm

    remove_algorithm_parameters(algorithm_name, algorithm_parameters)
            Description | Remove the parameters which were chosen by the user to a given algorithm

    set_similarity_score(similarity_list)
            Description | Set the list of all chosen similarity function by the user

    set_saving_model(save_model)
            Description | Set the variable which indicates whether the user want to save the current model or not

    run_models(algorithm, similarity_score, test_data_path, results_path, new_model_running)
            Description | Execute models creation/loading process

    set_new_model_running(new_model_running)
            Description | Set the variable which indicates whether the user chose to create new model or to load an
                          existing model

    set_existing_algorithms(algorithms_dict)
            Description | Set a dictionary which includes all the algorithm which were chosen by the user in a load
                          existing models flow

    set_features_columns_options()
            Description | Set the data set columns which were loaded from the test data set

    get_features_columns_options()
            Description | Get the data set columns which were loaded from the test data set

    set_users_selected_features(features_list, target_features_list)
            Description | Set the data set columns which were selected by the user

    add_new_thread(new_thread)
            Description | Add new running thread to the system

    get_existing_thread()
            Description | Get running thread

    get_new_model_running()
            Description | Indicator whether the user chose a new model creation flow or not

    set_current_algorithm_to_edit(algorithm_name)
            Description | Set the algorithm which the user is editing at a specific moment

    get_current_algorithm_to_edit()
            Description | Get the algorithm which the user is editing at a specific moment

    get_algorithms()
            Description | Get all the algorithms

    remove_algorithm(algorithm_name)
            Description | Remove a given algorithm

    set_results_selected_algorithm(selected_algorithm)
            Description | Set the variable which indicates which algorithm should be shown in the results table
                          at this moment

    set_results_selected_flight_route(selected_flight_route)
            Description | Set the variable which indicates which flight route should be shown in the results table
                          at this moment

    get_results_selected_algorithm()
            Description | Get the variable which indicates which algorithm should be shown in the results table
                          at this moment

    get_results_selected_flight_route()
            Description | Get the variable which indicates which flight route should be shown in the results table
                          at this moment

    reset_input_settings_params()
            Description | Reset all the values of input settings attributes

    get_flight_routes()
            Description | Get all the flight routes in the test set

    get_existing_algorithms()
            Description | Get a dictionary which includes all the algorithm which were chosen by the user in a load
                          existing models flow

    get_similarity_functions()
            Description | Get all similarity functions which were chosen by the user

    set_results_selected_similarity_function(similarity_function)
            Description | Set the variable which indicates which similarity function should be shown in the results
                          table at this moment

    get_results_selected_similarity_function()
            Description | Get the variable which indicates which similarity function should be shown in the results
                          table at this moment

    get_results_metrics_data()
            Description |  Get the dictionary which includes all the metrics for the current flow

    set_tune_model_input_path(input_path)
            Description | Set the path for data for tuning a model

    get_tune_model_input_path()
            Description | Get the path for data for tuning a model

    set_tune_model_features()
            Description | Set features list for tune model flow

    get_tune_model_features()
            Description | Get features list for tune model flow

     set_tune_model_configuration(input_features, target_features, window_size, algorithm)
         Description | Set full configuration for tune model flow

    get_tune_flow_input_features()
            Description | Get input features for tune model flow

    get_tune_flow_target_features()
            Description | Get target features for tune model flow

    get_tune_flow_window_size()
            Description | Get window sizes for tune model flow

    get_tune_flow_algorithm()
            Description | Get algorithm for tune model flow

    set_tune_model_results_path(input_path)
            Description | Set the path for results for tuning a model

    get_tune_model_results_path()
            Description | Get the path for results for tuning a model

    run_tuning()
            Description | Execute models tuning process

    get_window_size(algorithm)
            Description | Get the chosen window size for a specific algorithm

    init_models()
            Description | Init models dictionary

    """

    def __init__(self, *args, **kwargs):
        """
        GUI controller init function
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.model_controller = ModelController(self)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.geometry('800x600')

        # The container is where we'll stack a bunch of frames on top of each other,
        # then the one we want visible will be raised above the others
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Disables ability to tear menu bar into own window
        # container.option_add('*tearOff', 'FALSE')

        self.current_algorithm_to_edit = "LSTM"
        self.frames = {}
        self.init_main_controller()

    def init_main_controller(self):
        """
        Init main controller functionality
        :return:  all frames in the applciation
        """

        # Iterate over all existing UI windows
        for F in (MainWindow,
                  NewModel,
                  LoadModel,
                  PreTuneModel,
                  TuneModel,
                  AlgorithmsWindow,
                  FeatureSelectionWindow,
                  SimilarityFunctionsWindow,
                  ExistingAlgorithmsWindow,
                  LoadingWindow,
                  ResultsWindow,
                  ParametersOptionsWindow,
                  ResultsTableWindow,
                  TuningLoadingWindow,
                  LoadedDataWindow,
                  TuneResultsWindow,
                  ResultsPlotWindow):
            page_name = F.__name__

            # Init each frame
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame

            # Put all of the pages in the same location; the one on the top of the stacking
            # order will be the one that is visible
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainWindow")

    def show_frame(self, page_name):
        """
        Show the frame for a given page name
        :param page_name: input page
        :return: shown frame on the screen
        """

        frame = self.frames[page_name]
        frame.tkraise()

    def reinitialize_frame(self, page_name):
        """
        Reinitialize a frame for a given page name
        :param page_name:  input page
        :return: reinitialized frame
        """

        frame = self.frames[page_name]
        frame.reinitialize()
        frame.tkraise()

    def reset_frame(self):
        """
        Reset all existing frames
        :return: reset of all widgets in the system
        """

        for frame in self.frames.values():
            frame.reset_widgets()

    def set_new_model_training_input_path(self, input_path):
        self.model_controller.set_training_data_path(input_path)

    def set_new_model_test_input_path(self, input_path):
        self.model_controller.set_test_data_path(input_path)

    def set_new_model_results_input_path(self, input_path):
        self.model_controller.set_results_path(input_path)

    def set_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        return self.model_controller.set_algorithm_parameters(algorithm_name, algorithm_parameters)

    def remove_algorithm_parameters(self, algorithm_name, algorithm_parameters):
        self.model_controller.remove_algorithm_parameters(algorithm_name, algorithm_parameters)

    def set_similarity_score(self, similarity_list):
        self.model_controller.set_similarity_score(similarity_list)

    def set_saving_model(self, save_model):
        self.model_controller.set_saving_model(save_model)

    def run_models(self, algorithm, similarity_score, test_data_path, results_path, new_model_running, event):
        self.model_controller.run_models(algorithm, similarity_score, test_data_path,
                                         results_path, new_model_running, event)

    def set_new_model_running(self, new_model_running):
        self.model_controller.set_new_model_running(new_model_running)

    def set_existing_algorithms(self, algorithms_dict):
        self.model_controller.set_existing_algorithms(algorithms_dict)

    def set_features_columns_options(self):
        self.model_controller.set_features_columns_options()

    def get_features_columns_options(self):
        return self.model_controller.get_features_columns_options()

    def set_users_selected_features(self, features_list, target_features_list):
        self.model_controller.set_users_selected_features(features_list, target_features_list)

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

    def set_results_selected_algorithm(self, selected_algorithm):
        self.model_controller.set_results_selected_algorithm(selected_algorithm)

    def set_results_selected_flight_route(self, selected_flight_route):
        self.model_controller.set_results_selected_flight_route(selected_flight_route)

    def get_results_selected_algorithm(self):
        return self.model_controller.get_results_selected_algorithm()

    def get_results_selected_flight_route(self):
        return self.model_controller.get_results_selected_flight_route()

    def reset_input_settings_params(self):
        self.model_controller.reset_input_settings_params()

    def get_flight_routes(self):
        return self.model_controller.get_flight_routes()

    def get_existing_algorithms(self):
        return self.model_controller.get_existing_algorithms()

    def get_similarity_functions(self):
        return self.model_controller.get_similarity_functions()

    def set_results_selected_similarity_function(self, similarity_function):
        self.model_controller.set_results_selected_similarity_function(similarity_function)

    def get_results_selected_similarity_function(self):
        return self.model_controller.get_results_selected_similarity_function()

    def get_results_metrics_data(self):
        return self.model_controller.get_results_metrics_data()

    def set_tune_model_input_path(self, input_path):
        self.model_controller.set_tune_model_input_path(input_path)

    def get_tune_model_input_path(self):
        return self.model_controller.get_tune_model_input_path()

    def set_tune_model_features(self):
        self.model_controller.set_tune_model_features()

    def get_tune_model_features(self):
        return self.model_controller.get_tune_model_features()

    def set_tune_model_configuration(self, input_features, target_features, window_size, algorithm):
        self.model_controller.set_tune_model_configuration(input_features, target_features, window_size, algorithm)

    def get_tune_flow_input_features(self):
        return self.model_controller.get_tune_flow_input_features()

    def get_tune_flow_target_features(self):
        return self.model_controller.get_tune_flow_target_features()

    def get_tune_flow_window_size(self):
        return self.model_controller.get_tune_flow_window_size()

    def get_tune_flow_algorithm(self):
        return self.model_controller.get_tune_flow_algorithm()

    def set_tune_model_results_path(self, results_path):
        self.model_controller.set_tune_model_results_path_path(results_path)

    def get_tune_model_results_path_path(self):
        return self.model_controller.get_tune_model_results_path_path()

    def run_tuning(self):
        self.model_controller.run_tuning()

    def get_window_size(self, algorithm):
        return self.model_controller.get_window_size(algorithm)

    def init_models(self):
        return self.model_controller.init_models()


# Main loop of the Anomaly Detection application
if __name__ == "__main__":
    app = AnomalyDetectionGUI()
    app.mainloop()
