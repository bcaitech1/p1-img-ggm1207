"""
# Sample a float uniformly between -5.0 and -1.0
>>> "uniform": tune.uniform(-5, -1),

# Sample a float uniformly between 3.2 and 5.4,
# rounding to increments of 0.2
>>> "quniform": tune.quniform(3.2, 5.4, 0.2),

# Sample a float uniformly between 0.0001 and 0.01, while
# sampling in log space
>>> "loguniform": tune.loguniform(1e-4, 1e-2),

# Sample a float uniformly between 0.0001 and 0.1, while
# sampling in log space and rounding to increments of 0.0005
>>> "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-4),

# Sample a random float from a normal distribution with
# mean=10 and sd=2
>>> "randn": tune.randn(10, 2),

# Sample a random float from a normal distribution with
# mean=10 and sd=2, rounding to increments of 0.2
>>> "qrandn": tune.qrandn(10, 2, 0.2),

# Sample a integer uniformly between -9 (inclusive) and 15 (exclusive)
>>> "randint": tune.randint(-9, 15),

# Sample a integer uniformly between 1 (inclusive) and 10 (exclusive),
# while sampling in log space
>>> "lograndint": tune.lograndint(1, 10),

# Sample a random uniformly between -21 (inclusive) and 12 (inclusive (!))
# rounding to increments of 3 (includes 12)
>>> "qrandint": tune.qrandint(-21, 12, 3),

# Sample a integer uniformly between 1 (inclusive) and 10 (inclusive (!)),
# while sampling in log space and rounding to increments of 2
>>> "qlograndint": tune.qlograndint(1, 10, 2),

# Sample an option uniformly from the specified choices
>>> "choice": tune.choice(["a", "b", "c"]),

# Sample from a random function, in this case one that
# depends on another value from the search space
>>> "func": tune.sample_from(lambda spec: spec.config.uniform * 0.01),

# Do a grid search over these values. Every value will be sampled
# `num_samples` times (`num_samples` is the parameter you pass to `tune.run()`)
>>> "grid": tune.grid_search([32, 64, 128])
"""

from ray import tune

strat = dict()
strat["st01"] = {
    "strategy": "st01",
    "data_kind": "dataset_v1",
    "ms_name": "koelectra",
    #  "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
    "model_name_or_path": "bert-base-multilingual-cased",
    "batch_size": tune.choice([64]),
    "max_seq_length": 256,
    "weight_decay": tune.uniform(0.001, 0.02),
    "learning_rate": tune.uniform(3e-4, 3e-3),
    "optimizer": "adamw",
    "optimizer_hp": {"test": 3},
    "scheduler": "step_lr",
    "scheduler_hp": {"test": 3},
}
