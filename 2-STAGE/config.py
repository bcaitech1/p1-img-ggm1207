import argparse
import os.path as p

from functools import partial


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

    # ops hypterparameter: Uses when (not training)

    pa("--num_labels", default=42)
    pa("--save_steps", default=500)
    pa("--logging_steps", default=100)
    pa("--save_total_limit", default=3)
    pa("--logging_dir", default="./logs")
    pa("--output_dir", default="./results", type=str)
    pa("--model_name_or_path", default="bert-base-multilingual-cased")
    pa("--data_dir", default="/home/j-gunmo/storage/data/input/data/", type=str)

    # train hypterparameter: Uses when training

    pa("--warmup_steps", default=500)
    pa("--weight_decay", default=0.01)
    pa("--learning_rate", default=5e-5)
    pa("--num_train_epochs", default=4)
    pa("--evaluation_strategy", default="epoch")
    pa("--per_device_eval_batch_size", default=16)
    pa("--per_device_train_batch_size", default=16)

    # parsing
    args, unknown = parser.parse_known_args()

    return args
