import os
import time
import math

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from config import get_args
from network import ResNetClassification
from prepare import get_dataloader, get_classes


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(args, model, optimizer, loss_fn, dataloader):
    model.train()

    epoch_loss = 0.0
    label_idx = ["gender", "age", "mask"].index(args.train_key)

    for idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        labels = labels[label_idx]
        images, labels = images.float().to(args.device), labels.to(args.device)

        if args.train_key == "age":
            labels = labels.float().to(args.device)

        output = model(images)

        if args.train_key == "age":
            output = torch.squeeze(output, 1)

        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader):
    model.eval()

    epoch_loss = 0.0
    label_idx = ["gender", "age", "mask"].index(args.train_key)

    acc_count = 0
    acc_len = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            labels = labels[label_idx]
            images, labels = images.to(args.device), labels.to(args.device)

            output = model(images)

            if args.train_key != "age":
                acc_count += (
                    (labels.detach() == torch.argmax(output.detach(), dim=1))
                    .sum()
                    .item()
                )

                acc_len += len(labels)

            loss = loss_fn(output, labels)
            epoch_loss += loss.item()

    accuracy = 0

    if args.train_key != "age":
        accuracy = acc_count / acc_len

    return epoch_loss / len(dataloader), accuracy


def run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader):
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(args, model, optimizer, loss_fn, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, loss_fn, test_dataloader)

        if valid_loss < best_valid_loss:
            #  model_save_path = os.path.join(
            #      args.model_path, f"{wandb.run.name}-{args.train_key}.pt"
            #  )
            model_save_path = os.path.join(args.model_path, f"{args.train_key}.pt")
            best_valid_loss = valid_loss
            torch.save(model, model_save_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        wandb.log(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "epoch": epoch,
            }
        )

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tValidation Loss: {valid_loss:.3f}")
        print(f"\tValidation Accuracy: {valid_acc:.3f}")


def main(args):
    wandb.init(project="p-stage-1", reinit=True)
    wandb.config.update(args)

    args = wandb.config

    train_dataloader, test_dataloader = get_dataloader(args)

    classes = get_classes(args.train_key)
    args.classes = classes

    num_class = 1 if args.train_key == "age" else len(args.classes)

    model = ResNetClassification(num_class).to(args.device)
    model.apply(init_weights)

    wandb.watch(model)

    if wandb.config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif wandb.config.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    loss_fn = nn.MSELoss() if args.train_key == "age" else nn.CrossEntropyLoss()

    run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
