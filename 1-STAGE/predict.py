import os
from PIL import Image

import wandb
import torch
import pandas as pd
import torch.nn as nn

from train import evaluate, get_lossfn
from config import get_args
from prepare import get_dataloader, get_transforms
from metrics import (
    change_2d_to_1d,
    change_age_to_cat,
    cal_metrics,
    cal_accuracy,
    FocalLoss,
)


def eval_class(mi, gi, ai):
    return 6 * mi + 3 * gi + ai


def _log_scores_to_df(args, summary_table, key, model, loss_fn, data_loader):
    args.train_key = key
    _, labels, outputs = evaluate(args, model, loss_fn, data_loader)

    f1_score = cal_metrics(outputs, labels)
    acc_score = cal_accuracy(outputs, labels)

    summary_table.loc[key] = [f1_score, acc_score]

    return labels, outputs


def log_scores_to_df(args, summary_table, keys, models, data_loader):
    """ loss_fn: use same model """

    label_list, output_list = [], []

    for model, key in zip(models, keys):
        args.train_key = key
        loss_fn = get_lossfn(args)

        labels, outputs = _log_scores_to_df(
            args, summary_table, key, model, loss_fn, data_loader
        )

        label_list.append(labels.detach().cpu().numpy())
        output_list.append(outputs.detach().cpu().numpy())

    return label_list, output_list


def main(args):
    print("".join([f"{k:<15} : {v}\n" for k, v in sorted(args.items(), key=len)]))

    wandb.init(project="p-stage-1", reinit=True)
    wandb.config.update(args)
    wandb.run.name = f"predict-{wandb.run.name}"

    try:
        age_model = torch.load(args.age_model)
        gender_model = torch.load(args.gender_model)
        mask_model = torch.load(args.mask_model)
    except Exception as e:
        print(e)
        raise "Failed Model load"

    age_model.eval()
    gender_model.eval()
    mask_model.eval()

    _, valid_dataloader = get_dataloader(args)
    transform = get_transforms(args)

    eval_dir = "/opt/ml/input/data/eval/"
    eval_df = pd.read_csv(os.path.join(eval_dir, "info.csv"))

    summary_table = pd.DataFrame(columns=["f1_score", "accuracy"])

    keys = ["mask", "gender", "age"]
    models = [mask_model, gender_model, age_model]

    # mga: mask, gender, age (sequence)
    mga_label_lists, mga_output_lists = log_scores_to_df(
        args, summary_table, keys, models, valid_dataloader
    )

    labels, outputs = [], []

    for mi, gi, ai in zip(mga_label_lists):
        labels.append(eval_class(mi, gi, ai))

    for mi, gi, ai in zip(mga_output_lists):
        outputs.append(eval_class(mi, gi, ai))

    acc = cal_accuracy(torch.tensor(labels), torch.tensor(outputs))
    table = wandb.Table(dataframe=summary_table, rows=keys)

    wandb.log({"Result": table, "valid_accuracy": acc})


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
