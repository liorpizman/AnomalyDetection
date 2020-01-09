import tk as tk

from utils.shared.input_settings import input_settings


class results_window(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        # controller.geometry('800x600')
        # # Create Widgets
        #
        #
        # # do it in function
        # self.algorithms = input_settings.get_algorithms()
        # self.similarity_score = input_settings.get_similarity()
        # self.results_path = input_settings.get_results_path()
        # self.save_model = input_settings.get_saving_model()
        #
        # self.results_window_title = tk.Label(self, text="Results", font=controller.title_font)
        #
        # self.results = {}
        #
        # for algorithm in self.algorithms:
        #     self.results[algorithm] = {}
        #     for similarity in self.similarity_score:
        #
        # self.training_label = tk.Label(self, text="Training directory")
        # self.training_input = tk.Entry(self, width=80)
        # self.training_btn = tk.Button(self, text="Browse", command=self.set_input_path)
        #
        # # Layout using grid
        # self.new_model_title.grid(row=0, column=1, pady=3)
        #
        # self.training_label.grid(row=1, column=0, pady=3)
        # self.training_input.grid(row=1, column=1, pady=3, padx=10)
        # self.training_btn.grid(row=1, column=2, pady=3)
        #
        # self.test_input.grid(row=2, column=1, pady=3, padx=10)
        # self.test_label.grid(row=2, column=0, pady=3)
        # self.test_btn.grid(row=2, column=2, pady=3)
        #
        # self.results_input.grid(row=3, column=1, pady=3, padx=10)
        # self.results_label.grid(row=3, column=0, pady=3)
        # self.results_btn.grid(row=3, column=2, pady=3)
        #
        # self.back_button.grid(row=15, column=0, pady=3)
        # self.next_button.grid(row=15, column=3, pady=3)