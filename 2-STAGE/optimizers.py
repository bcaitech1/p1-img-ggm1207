import torch
import torch.optim as optim
from torch.utils.data import RandomSampler

OPTIMIZERS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.ASGD}

SCHEDULER = {
    "cyclic_lr": optim.lr_scheduler.CyclicLR,
    "cosine_lr": optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warmup_lr": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "step_lr": optim.lr_scheduler.StepLR,
}

SAMPLER = {"random": RandomSampler}


def get_optimizer(args, model):
    if args.optimizer not in OPTIMIZERS.keys():
        raise KeyError(f"{args.optimizer} not in OPTIMIZERS")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return OPTIMIZERS[args.optimizer](
        optimizer_grouped_parameters, lr=args.learning_rate
    )


def get_scheduler(args, optimizer):
    if args.scheduler not in SCHEDULER.keys():
        raise KeyError(f"{args.scheduler} not in SCHEDULER")

    return SCHEDULER[args.scheduler](optimizer, step_size=2)


def get_sampler(args):
    if args.sampler not in SAMPLER.keys():
        raise KeyError(f"{args.sampler} not in SAMPLER")

    return SAMPLER[args.sampler]
