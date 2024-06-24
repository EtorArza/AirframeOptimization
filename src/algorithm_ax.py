import numpy as np
import threading
import queue
from interfaces import *
from ax import optimize
from contextlib import contextmanager
import random
import numpy as np
import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


def restore_global_rs(old_state: dict):
    random.setstate(old_state["random"])
    np.random.set_state(old_state["numpy"])
    torch.set_rng_state(old_state["torch"])


def set_global_rs(seed: int) -> dict:
    old_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return old_state


# https://ax.dev/docs/bayesopt.html
class ax_optimizer:

    def __init__(self, prob:problem, seed: int, total_budget: int):
        self.prob = prob
        self.rs = np.random.RandomState(seed+78)
        self.total_budget = total_budget


        def fun(x):
            # numpy array for x from wierd framework format
            restore_global_rs(self.old_state)
            self.x_queue.put(np.array([el[1] for el in sorted(x.items(), key=lambda z: float(z[0].strip("x")))]))
            f_res = self.f_queue.get(timeout=30000.0)
            assert type(f_res)==float or type(f_res) == np.float64, "f_res = "+str(f_res) + "| type = " + str(type(f_res))
            set_global_rs(self.rs.randint(int(4294967293)))
            return f_res

        self.x_queue = queue.Queue()
        self.f_queue = queue.Queue()


        gs = GenerationStrategy(
            steps=[
                # 1. Initialization step (does not require pre-existing data and is well-suited for
                # initial sampling of the search space)
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=100,  # How many trials should be produced from this generation step
                    min_trials_observed=100,  # How many trials need to be completed to move to next model
                    max_parallelism=1,  # Max parallelism for this step
                    model_kwargs={"seed": seed},  # Any kwargs you want passed into the model
                    model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                ),
                # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                # from all data available at the time of each new candidate generation call)
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,  # No limitation on how many trials should be produced from this step
                    max_parallelism=1,  # Parallelism limit for this step, often lower than for Sobol
                    # More on parallelism vs. required samples in BayesOpt:
                    # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
                ),
            ]
        )


        def minimize():
            best_parameters, best_values, experiment, model = optimize(
                parameters=[
                {
                    "name": f"x{i}",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.0, 1.0],
                } for i in range(prob.dim)
                ],
                evaluation_function=fun,
                minimize=True,
                total_trials=total_budget,
                generation_strategy=gs,

            )
        self.old_state = set_global_rs(self.rs.randint(4294967293))
        thread = threading.Thread(target=minimize, args=[], daemon=True)
        thread.start()

        # print(best_parameters)
        # param.random_state = self.rs

    def ask(self):
        return self.x_queue.get()

    def tell(self, f):
        self.f_queue.put(f)

