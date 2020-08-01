from src.running_instance import RunningInstance


class RandomSampling(RunningInstance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self):
        return self.select_random(self.train_idx, self.experiment.model.rng)
