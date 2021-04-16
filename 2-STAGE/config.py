import argparse
from functools import partial

import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="2-stage hypterparameter")
    pa = partial(parser.add_argument)

    # pathes
    pa(
        "--weight_dir",
        default="/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/weights",
    )
    pa("--data_dir", default="/home/j-gunmo/storage/data/input/data/")
    pa("--data_kind", default="dataset_v1")
    pa("--strategy", default="st01")
    pa(
        "--submit_dir",
        default="/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/submits",
    )

    #  # server pathes
    #  pa("--weight_dir", default="/opt/ml/P-Stage/2-STAGE/weights")
    #  pa("--data_dir", default="/opt/ml/input/data")
    #  pa("--data_kind", default="dataset_v1")
    #  pa("--strategy", default="st01")
    #  pa("--submit_dir", default="/opt/ml/P-Stage/2-STAGE/submits")

    # ops hypterparameter: Uses when (not training)

    pa("--dataset_idx", default=0)
    pa("--num_labels", default=42, type=int)
    pa("--auto_sub", default=False, type=str2bool)
    pa("--debug", default=False, type=str2bool)

    # early stopping
    pa("--delta", default=0, type=int)
    pa("--patience", default=5, type=int)

    # train hypterparameters: Uses when training

    #  pa("--ms_name", default="bert")
    #  pa("--model_name_or_path", default="bert-base-multilingual-cased")
    pa("--ms_name", default="koelectra")
    pa("--model_name_or_path", default="monologg/koelectra-base-discriminator")

    pa("--optimizer", default="adamw")
    pa("--sampler", default="random")
    pa("--epochs", default=20, type=int)
    pa("--scheduler", default="sgdr")
    pa("--batch_size", default=16, type=int)
    pa("--warmup_steps", default=500, type=int)
    pa("--weight_decay", default=0.01, type=float)
    #  pa("--learning_rate", default=5e-5, type=float)
    pa("--max_seq_length", default=128, type=int)

    pa("--optimizer_hp", default={"lr": 5e-5}, type=dict)
    pa(
        "--scheduler_hp",
        default={
            "first_cycle_steps": 200,
            "cycle_mult": 1.0,
            "warmup_steps": 50,
            "gamma": 0.5,
        },
        type=dict,
    )

    # parsing
    args, unknown = parser.parse_known_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return args
