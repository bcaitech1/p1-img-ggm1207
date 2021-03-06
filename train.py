import os
import time
import json
import math
import pickle
import random
import warnings
import functools
from datetime import datetime

import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from coral_pytorch.dataset import proba_to_label, levels_from_labelbatch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from config import get_args
from predict import get_all_datas
from network import ResNetClassification, get_resnet34
from prepare import get_dataloader, get_classes, get_num_classes
from log_helper import log_confusion_matrix_by_images, log_confusion_matrix

from metrics import (
    get_lossfn,
    get_optimizers,
    change_2d_to_1d,
    tensor_to_numpy,
    apply_grad_cam_pp_to_images,
    tensor_images_to_numpy_images,
)

from log_helper import plots_result

warnings.filterwarnings(action="ignore")


def get_label_fn(args):

    if args.loss_metric == "coral_loss":

        def _get_label_fn(preds):
            probas = torch.sigmoid(preds)
            labels = proba_to_label(probas)
            return labels

    else:

        def _get_label_fn(preds):
            labels = torch.argmax(preds, dim=1)
            return labels

    return _get_label_fn


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_loss(args, loss_fn, outputs, labels):
    return loss_fn(outputs, labels)


def train(args, model, optimizer, scheduler, scaler, loss_fn, dataloader):
    model.train()
    epoch_loss = 0.0

    num_class = get_num_classes(args)

    for idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        if args.loss_metric == "coral_loss":
            labels = levels_from_labelbatch(labels, num_classes=num_class)

        images, labels = images.to(args.device), labels.to(args.device)

        with autocast():
            outputs = model(images)
            loss = get_loss(args, loss_fn, outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #  optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader):
    model.eval()

    epoch_loss = 0.0

    all_labels = torch.tensor([]).to(args.device)
    all_preds = torch.tensor([]).to(args.device)

    get_labels = get_label_fn(args)
    num_class = get_num_classes(args)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            all_labels = torch.cat((all_labels, labels.to(args.device)))

            if args.loss_metric == "coral_loss":
                labels = levels_from_labelbatch(labels, num_classes=num_class)

            images, labels = images.to(args.device), labels.to(args.device)
            preds = model(images)

            loss = get_loss(args, loss_fn, preds, labels)
            epoch_loss += loss.item()

            #  preds = torch.argmax(preds, dim=1)
            preds = get_labels(preds)
            preds = change_2d_to_1d(preds)

            all_preds = torch.cat((all_preds, preds))

    return epoch_loss / len(dataloader), all_labels, all_preds


def run(
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    train_dataloader,
    test_dataloader,
):
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(
            args, model, optimizer, scheduler, scaler, loss_fn, train_dataloader
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")

        if args.eval:

            valid_loss, label_list, output_list = evaluate(
                args, model, loss_fn, test_dataloader
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_save_path = os.path.join(args.model_path, f"{wandb.run.name}.pt")
                torch.save(model, model_save_path)

                with open(model_save_path[:-2] + "args", "w") as f:
                    json.dump(dict(args), f)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            output_list = tensor_to_numpy(output_list)
            label_list = tensor_to_numpy(label_list)

            f1_sco = f1_score(output_list, label_list, average="macro")
            pr_sco = precision_score(output_list, label_list, average="macro")
            re_sco = recall_score(output_list, label_list, average="macro")
            ac_sco = accuracy_score(output_list, label_list)

            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_f1_score": f1_sco,
                    "valid_pr_score": pr_sco,
                    "valid_re_score": re_sco,
                    "valid_accuracy": ac_sco,
                    "epoch": epoch,
                }
            )

            print(f"\tValidation Loss: {valid_loss:.3f}")
            print(f"\tValidation f1 score: {f1_sco:.3f}")
            print(f"\tValidation Acc: {ac_sco:.3f}")
            print()

    # Last Visualization
    if not args.eval:
        model_save_path = os.path.join(args.model_path, f"{wandb.run.name}.pt")
        torch.save(model, model_save_path)
        return

    model = torch.load(model_save_path).to(args.device)

    _, labels, preds = evaluate(args, model, loss_fn, test_dataloader)

    labels = tensor_to_numpy(labels)
    preds = tensor_to_numpy(preds)

    fig = log_confusion_matrix(args, labels, preds)
    wandb.log({"Valid Confusion Matirx": fig})

    example_images = []
    sup_titles = ["TRAIN", "VALIDATION"]

    get_labels = get_label_fn(args)

    for idx, dataloader in enumerate([train_dataloader, test_dataloader]):
        images, labels = next(iter(dataloader))
        images, labels = images.to(args.device), labels.to(args.device)

        preds = model(images)

        images = apply_grad_cam_pp_to_images(
            args, model, images
        )  # return same shape tensor
        images = tensor_images_to_numpy_images(images, renormalize=False)
        labels = tensor_to_numpy(labels)

        preds = tensor_to_numpy(preds)

        #  fig = plots_result(args, images, labels, preds, sup_titles[idx])
        #  example_images.append(wandb.Image(fig))

    wandb.log({"Traininig Visualization": example_images})


def main(args):
    wandb.init(project="p-stage-1", reinit=True)
    wandb.config.update(args)

    args = wandb.config

    wandb.run.name = (
        f"{args.train_key}-{datetime.now().strftime('%m%d%H%M')}-{wandb.run.name}"
    )

    train_dataloader, test_dataloader = get_dataloader(args)

    num_class = get_num_classes(args)

    #  model = ResNetClassification(args, num_class).to(args.device)
    model = get_resnet34(args, num_class).to(args.device)
    model.apply(init_weights)
    wandb.watch(model)

    print("wandb.config:")
    print("".join([f"{k:<15} : {v}\n" for k, v in sorted(args.items(), key=len)]))

    optimizer = get_optimizers(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=0
    )
    scaler = GradScaler()

    loss_fn = get_lossfn(args)

    run(
        args,
        model,
        optimizer,
        scheduler,
        scaler,
        loss_fn,
        train_dataloader,
        test_dataloader,
    )


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU

    # ?????? ?????? ????????? ??????????????? ??????.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
