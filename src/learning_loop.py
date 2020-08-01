from tqdm import tqdm
from src.utils.constants import METHODS
from src.utils.common import extract_param_from_name


class LearningLoop:

    def __init__(self):
        self.initial_known_idx = []
        self.initial_train_idx = []
        self.initial_test_idx = []

    def update_data_split(self, known_idx, train_idx, test_idx):
        self.initial_known_idx = known_idx
        self.initial_train_idx = train_idx
        self.initial_test_idx = test_idx

    def run(self, args, experiment, method):
        """
        Perform the learning loop: train a model, select a query and retrain the model until the budget is exhausted.

        :param args: The input arguments
        :param experiment: The experiment being performed
        :param method: The method to be used for query selection
        """
        running_instance = self.initialize_running_instance(args, experiment, method)
        running_instance.train()

        for iteration in tqdm(range(args.max_iter)):
            running_instance.iteration = iteration
            # If we have selected all instances in the train dataset
            if len(running_instance.train_idx) <= 1:
                break

            # 1. Query
            query_idx = running_instance.query()
            if query_idx is None:
                continue

            running_instance.calculate_query_accuracy(query_idx)
            running_instance.plot(title="{} {}".format(iteration, method), query_idx=query_idx)
            # 2. Label
            running_instance.label_query(query_idx)
            # 3. Retrain
            running_instance.train()

        running_instance.plot(title="{} {}".format("Last iteration", method))

        return running_instance

    def initialize_running_instance(self, args, experiment, method):
        param, method = extract_param_from_name(method)

        return METHODS[method](
            args=args,
            experiment=experiment,
            known_idx=self.initial_known_idx,
            train_idx=self.initial_train_idx,
            test_idx=self.initial_test_idx,
            param=param,
        )
