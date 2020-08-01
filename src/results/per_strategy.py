class ResultsPerStrategy:
    
    def __init__(self):
        self.scores_f1 = []
        self.test_scores_f1 = []
        self.scores_auc = []
        self.test_scores_auc = []
        self.query_scores = []
        self.check_rules_f1 = []
        self.false_mistakes_count = [0]