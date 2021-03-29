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

# TODO: uniform 분포로 수정 해야 됨
#  def weights_init(m):
#      classname = m.__class__.__name__
#      if classname.find("Conv") != -1:  # -1 mean, no find
#          nn.init.normal_(m.weight, 0.0, 0.02)
#      elif classname.find("BatchNorm") != -1:
#          nn.init.normal_(m.weight, 1.0, 0.02)
#          nn.init.zeros_(m.bias)


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
        images, labels = images.to(args.device), labels.to(args.device)

        output = model(images)

        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader):
    model.eval()

    epoch_loss = 0.0
    label_idx = ["gender", "age", "mask"].index(args.train_key)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            labels = labels[label_idx]
            images, labels = images.to(args.device), labels.to(args.device)

            output = model(images)

            loss = loss_fn(output, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader):
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train(args, model, optimizer, loss_fn, train_dataloader)
        valid_loss = evaluate(args, model, loss_fn, test_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss >> d_loss : {d_loss:.3f} g_loss: {g_loss:.3f}")
        print(
            f"\tValidation Loss >> d_loss : {d_valid_loss:.3f} g_loss: {g_valid_loss:.3f}"
        )


def main(args):
    train_dataloader, test_dataloader = get_dataloader(args)

    classes = get_classes(args.train_key)
    args.classes = classes

    model = ResNetClassification(len(args.classes)).to(args.device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() if args.train_key == "age" else nn.CrossEntropyLoss()

    run(args, model, optimizer, loss_fn, train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
