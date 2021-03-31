import os
import time
import math
import random
from datetime import datetime

import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from config import get_args
from network import ResNetClassification
from prepare import get_dataloader, get_classes

from metrics import cal_metrics, cal_accuracy, change_2d_to_1d, FocalLoss
from log_helper import plots_result


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_loss(args, loss_fn, outputs, labels):
    loss = loss_fn(outputs, labels)
    return loss


def get_optimizers(args, model):
    optim_fn = optim.Adam
    if args.optimizer == "adamw":
        optim_fn = optim.AdamW
    if args.optimizer == "sgd":
        optim_fn = optim.SGD
    return optim_fn(model.parameters(), lr=args.lr)


def get_lossfn(args):
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss(gamma=2)
    return loss_fn


def train(args, model, optimizer, loss_fn, dataloader):
    model.train()
    epoch_loss = 0.0

    for idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images, labels = images.to(args.device), labels.to(args.device)

        outputs = model(images)
        loss = get_loss(args, loss_fn, outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader):
    model.eval()

    epoch_loss = 0.0
    label_list = torch.tensor([]).to(args.device)
    output_list = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)

            outputs = model(images)

            loss = get_loss(args, loss_fn, outputs, labels)
            epoch_loss += loss.item()

            outputs = torch.argmax(outputs, dim=1)
            outputs = change_2d_to_1d(outputs)

            label_list = torch.cat((label_list, labels))
            output_list = torch.cat((output_list, outputs))

    return epoch_loss / len(dataloader), label_list, output_list


def run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader):
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(args, model, optimizer, loss_fn, train_dataloader)
        valid_loss, label_list, output_list = evaluate(
            args, model, loss_fn, test_dataloader
        )

        if valid_loss < best_valid_loss:
            model_save_path = os.path.join(args.model_path, f"{wandb.run.name}.pt")
            best_valid_loss = valid_loss
            torch.save(model, model_save_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        f1_sco = cal_metrics(output_list, label_list)
        acc = cal_accuracy(output_list, label_list)

        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_f1_score": f1_sco,
                "valid_accuracy": acc,
                "epoch": epoch,
            }
        )

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tValidation Loss: {valid_loss:.3f}")

    train_images, train_labels = next(iter(train_dataloader))
    test_images, test_labels = next(iter(test_dataloader))

    train_images, train_labels = train_images.to(args.device), train_labels.to(args.device)
    test_images, test_labels = test_images.to(args.device), test_labels.to(args.device)

    train_outputs = model(train_images)
    test_outputs = model(test_images)

    fig1 = plots_result(args, train_images, train_outputs, train_labels)
    fig2 = plots_result(args, test_images, test_outputs, test_labels)

    wandb.log({"train result": fig1, "test result": fig2})


def main(args):
    wandb.init(project="p-stage-1", reinit=True)
    wandb.config.update(args)
    wandb.run.name = f"{args.train_key}-{datetime.now().strftime('%m%d%H%M')}-{wandb.run.name}"
    args = wandb.config

    train_dataloader, test_dataloader = get_dataloader(args)

    num_class = len(get_classes(args.train_key))

    model = ResNetClassification(num_class).to(args.device)
    model.apply(init_weights)
    wandb.watch(model)

    print("wandb.config:")
    print("".join([f"{k:<15} : {v}\n" for k, v in sorted(args.items(), key=len)]))

    optimizer = get_optimizers(args, model)
    loss_fn = get_lossfn(args)

    run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
