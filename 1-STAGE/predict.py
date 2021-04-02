import os
from PIL import Image

import wandb
import torch
import pandas as pd
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)

from config import get_args
from train import get_lossfn
from prepare import get_dataloader, get_transforms, get_classes
from metrics import (
    FocalLoss,
    change_2d_to_1d,
    calulate_18class,
    tensor_to_numpy,
    tensor_images_to_numpy_images,
)
from log_helper import log_f1_and_acc_scores, log_confusion_matrix


def predict(args, model, dataloader):
    model.eval()

    all_images = torch.tensor([]).to(args.device)
    all_labels = torch.tensor([]).to(args.device)
    all_preds = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)

            preds = model(images)
            preds = torch.argmax(preds, dim=1)
            preds = change_2d_to_1d(preds)

            all_images = torch.cat((all_images, images))
            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, preds))

    return all_images, all_labels, all_preds


def predict_and_logs_by_class_with_all_models(args, keys, models):
    """ loss_fn: use same model """

    final_zip_labels, final_zip_preds = [], []
    loss_fn = get_lossfn(args).to(args.device)

    summary_table = pd.DataFrame([])

    for model, key in zip(models, keys):
        # mask, age, gender
        args.train_key = key
        _, valid_dataloader = get_dataloader(args)

        all_images, all_labels, all_preds = predict(args, model, valid_dataloader)

        all_images = tensor_images_to_numpy_images(all_images)
        all_labels = tensor_to_numpy(all_labels)
        all_preds = tensor_to_numpy(all_preds)

        log_f1_and_acc_scores(args, summary_table, all_labels, all_preds)
        fig = log_confusion_matrix(args, all_labels, all_preds)

        final_zip_labels.append(all_labels)
        final_zip_preds.append(all_preds)

    return summary_table, final_zip_labels, final_zip_preds


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
        labels.append(calulate_18class(mi, gi, ai))

    for (mi, gi, ai) in zip(*mga_output_lists):
        outputs.append(calulate_18class(mi, gi, ai))

    acc = cal_accuracy(torch.tensor(labels), torch.tensor(outputs))
    table = wandb.Table(dataframe=summary_table, rows=keys)

    wandb.log({"Result": table, "valid_accuracy": acc})


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
