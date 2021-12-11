import random
from typing import Tuple

from neps.optimizers.optimizer import Optimizer

from ..bayesian_optimization.acquisition_function_optimization.random_sampler import (
    RandomSampler,
)


class RandomSearch(Optimizer):
    def __init__(self, pipeline_space, return_opt_details: bool = False):
        super().__init__()
        self.sampled_idx = []
        self.return_opt_details = return_opt_details
        self.surrogate_model = None
        self.random_sampler = RandomSampler(pipeline_space)

    def initialize_model(self, **kwargs):
        pass

    def update_model(self, **kwargs):
        pass

    def propose_new_location(
        self, batch_size: int = 5, n_candidates: int = 10
    ) -> Tuple[Tuple, dict]:
        # create candidate pool
        pool = self.random_sampler.sample(n_candidates)

        next_x = random.sample(pool, batch_size)
        self.sampled_idx.append(next_x)

        opt_details = {"pool": pool} if self.return_opt_details else None

        return next_x, opt_details

    def get_config(self):
        config = self.random_sampler.sample(1)[0]

        return config

    def new_result(self, job):
        # if job.result is None:
        #     loss = np.inf
        # else:
        #     loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        #
        # config = job.kwargs["config"]
        pass