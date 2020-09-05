from collections import defaultdict
from src.utils import dump, get_passive_score


class Results:
    def __init__(self):
        self.scores_dict_f1 = defaultdict(list)
        self.scores_test_dict_f1 = defaultdict(list)
        self.scores_dict_f1_weighted = defaultdict(list)
        self.scores_test_dict_f1_weighted = defaultdict(list)
        self.scores_dict_auc = defaultdict(list)
        self.scores_test_dict_auc = defaultdict(list)
        self.scores_queries_dict_f1 = defaultdict(list)
        self.false_mistakes_dict = defaultdict(list)
        self.rules_wrt_blackbox_f1_dict = defaultdict(list)
        self.percent_uus_dict = defaultdict(list)
        self.score_passive_f1 = []
        self.score_passive_auc = []
        self.args = {}

    def collect(self, strategy, running_instance):
        res = running_instance.results
        self.scores_dict_f1[strategy].append(res.scores_f1)
        self.scores_test_dict_f1[strategy].append(res.test_scores_f1)
        self.scores_dict_f1_weighted[strategy].append(res.scores_f1_weighted)
        self.scores_test_dict_f1_weighted[strategy].append(res.test_scores_f1_weighted)
        self.scores_dict_auc[strategy].append(res.scores_auc)
        self.scores_test_dict_auc[strategy].append(res.test_scores_auc)
        self.scores_queries_dict_f1[strategy].append(res.query_scores)
        self.false_mistakes_dict[strategy].append(res.false_mistakes_count)
        self.rules_wrt_blackbox_f1_dict[strategy].append(res.rules_wrt_blackbox_f1)
        self.percent_uus_dict[strategy].append(res.percent_uus)
        self.score_passive_f1 = get_passive_score(running_instance, "f1_macro")
        self.score_passive_auc = get_passive_score(running_instance, "roc_auc")
        self.args = {
            'n_folds': running_instance.args.n_folds,
            'max_iter': running_instance.args.max_iter,
            'model_name': running_instance.experiment.model.name,
            'annotated_point': running_instance.annotated_point,
        }

    def export(self, path):
        dump(path + '__scores.pickle', {
            'train_f1': self.scores_dict_f1,
            'test_f1': self.scores_test_dict_f1,
            'train_f1_weighted': self.scores_dict_f1_weighted,
            'test_f1_weighted': self.scores_test_dict_f1_weighted,
            'train_auc': self.scores_dict_auc,
            'test_auc': self.scores_test_dict_auc,
            'queries_f1': self.scores_queries_dict_f1,
            'rules_wrt_blackbox_f1': self.rules_wrt_blackbox_f1_dict,
            'score_passive_f1': self.score_passive_f1,
            'score_passive_auc': self.score_passive_auc,
            'false_mistakes': self.false_mistakes_dict,
            'percent_uus': self.percent_uus_dict,
            'args': self.args,
        })
