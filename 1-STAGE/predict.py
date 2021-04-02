""" 모델의 다양한 성능을 측정  """

import os
from PIL import Image

import wandb
import torch
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from train import evaluate, get_lossfn
from config import get_args
from prepare import get_dataloader, get_transforms, get_classes
from metrics import (
    change_2d_to_1d,
    change_age_to_cat,
    cal_metrics,
    cal_accuracy,
    FocalLoss,
)


def eval_class(mi, gi, ai):
    return 6 * mi + 3 * gi + ai


def _log_f1_and_acc_scores(args, summary_table, labels, outputs):
    # class 별 f1_score를 계산해야함.

    classes = get_classes(args)
    
    for class_idx in range(len(classes)):
        fancy_index = np.where(labels == class_idx)

        f1 = f1_score(labels[fancy_idx], outputs[fancy_idx], average='macro')
        pr = precision_score(labels[fancy_idx], outputs[fancy_idx], average='macro')
        re = recall_score(labels[fancy_idx], outputs[fancy_idx], average='macro')
        acc = accuracy_score(labels[fancy_idx], outputs[fancy_idx])
        
        summary_table.loc[args.train_key, f"{class_idx} f1"] = f1
        summary_table.loc[args.train_key, f"{class_idx} pr"] = pr
        summary_table.loc[args.train_key, f"{class_idx} re"] = re
        summary_table.loc[args.train_key, f"{class_idx} acc"] = acc

def _log_confusion_matrix(args, labels, outputs):
    classes = get_classes(args)

    cf_matrix = confusion_matrix(outputs, labels)
    row_sums = 
     

    return


def _log_():
    return


def log_scores(args, keys, models):
    """ loss_fn: use same model """

    label_list, output_list = [], []
    loss_fn = get_lossfn(args).to(args.device)

    summary_table = pd.DataFrame([])

    for model, key in zip(models, keys):
        args.train_key = key
        _, valid_dataloader = get_dataloader(args)

        labels, outputs = evaluate(args, model, loss_fn, valid_dataloader)
        labels, outputs = labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()

        _log_f1_and_acc_scores(args, summary_table, labels, outputs)
        _log_confusion_matrix(args, labels, outputs)

        label_list.append(labels.detach().cpu().numpy())
        output_list.append(outputs.detach().cpu().numpy())

    return summary_table, label_list, output_list


def load_models(args):
    try:
        age_model = torch.load(args.age_model).to(args.device)
        gender_model = torch.load(args.gender_model).to(args.device)
        mask_model = torch.load(args.mask_model).to(args.device)

        age_model.eval()
        gender_model.eval()
        mask_model.eval()
    except Exception as e:
        raise e

    return [mask_model, gender_model, age_model]  # 순서 중요


def main(args):
    wandb.init(project="p-stage-1", reinit=True)
    wandb.config.update(args)
    wandb.run.name = f"predict-{wandb.run.name}"

    print(
        "".join([f"{k:<15} : {v}\n" for k, v in sorted(wandb.config.items(), key=len)])
    )

    models = load_models(args)
    keys = ["mask", "gender", "age"]

    # mga: mask, gender, age (sequence)
    summary_table, mga_label_lists, mga_output_lists = log_scores(
        args, summary_table, keys, models
    )

    labels, outputs = [], []

    for (mi, gi, ai) in zip(*mga_label_lists):
        labels.append(eval_class(mi, gi, ai))

    for (mi, gi, ai) in zip(*mga_output_lists):
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
