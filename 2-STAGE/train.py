import time
import traceback
from argparse import Namespace

import torch
import wandb
from sklearn.metrics import accuracy_score

from config import get_args
from losses import get_lossfn
from prepare import load_sample, load_dataloader
from database import execute_query
from slack import hook_fail_strategy
from networks import load_model_and_tokenizer
from utils import EarlyStopping, get_auto_save_path
from optimizers import get_optimizer, get_scheduler


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(args, model, loss_fn, optimizer, scheduler, dataloader):
    if isinstance(args, dict):
        args = Namespace(**args)

    model.train()
    epoch_loss = 0.0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        #  model.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"].to(args.device),
            "attention_mask": batch["attention_mask"].to(args.device),
            "token_type_ids": batch["token_type_ids"].to(args.device),
        }

        labels = batch["labels"].to(args.device)

        preds = model(**inputs, return_dict=True)
        loss = loss_fn(preds.logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader, return_keys=["loss", "acc"]):
    """ evaluate model and return dict of return_keys's results  """
    if isinstance(args, dict):
        args = Namespace(**args)

    model.eval()
    epoch_loss = 0.0

    results = dict()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
            }

            labels = batch["labels"].to(args.device)

            preds = model(**inputs, return_dict=True)

            if "loss" in return_keys:
                loss = loss_fn(preds.logits, labels)
                epoch_loss += loss.item()

            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.logits.detach().argmax(-1).cpu().tolist())

    assert len(all_labels) == len(all_preds)

    if "loss" in return_keys:
        results["loss"] = epoch_loss / len(dataloader)

    if "acc" in return_keys:
        results["acc"] = accuracy_score(all_labels, all_preds)

    if "preds" in return_keys:
        results["preds"] = all_preds

    return results


def run(args, model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader):
    """ train, evaluate for range(epochs), no hyperparameter search """
    args.save_path, _ = get_auto_save_path(args)
    early_stop = EarlyStopping(args, verbose=True)

    if isinstance(args, dict):
        args = Namespace(**args)

    for epoch in range(int(args.epochs)):
        start_time = time.time()

        train_loss = train(args, model, loss_fn, optimizer, scheduler, train_dataloader)
        results = evaluate(
            args, model, loss_fn, valid_dataloader, return_keys=["loss", "acc"]
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        early_stop(train_loss, results["loss"], results["acc"], model)

        if early_stop.early_stop is True:
            break

        wandb.log(
            dict(
                valid_loss=results["loss"],
                valid_acc=results["acc"],
                train_loss=train_loss,
                learning_rate=scheduler.get_last_lr()[0],
            )
        )

        print()


def debug(args, strategy):
    """ just testing, is it works? """

    if isinstance(args, dict):
        args = Namespace(**args)

    args.batch_size = 32
    args.max_seq_length = 128

    try:
        model, tokenizer = load_model_and_tokenizer(args)
        loss_fn = get_lossfn(args)
        optimizer = get_optimizer(args, model)
        _ = get_scheduler(args, optimizer)

        inputs, labels = load_sample(args, tokenizer)
        preds = model(**inputs, return_dict=True)
        _ = loss_fn(preds.logits, labels)

        query = f"UPDATE STRATEGY SET STATUS = 'RUN' WHERE strategy='{strategy}'"
        execute_query(query)

    except Exception:
        err_message = traceback.format_exc()
        print(err_message)
        query = f"UPDATE STRATEGY SET STATUS = 'PENDING' WHERE strategy='{strategy}'"
        execute_query(query)
        hook_fail_strategy(strategy, err_message)


if __name__ == "__main__":

    wandb.init(project="p-stage-2")

    args = get_args()

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    wandb.watch(model)
    wandb.config.update(args)

    train_dataloader, valid_dataloader = load_dataloader(args, tokenizer)
    loss_fn = get_lossfn(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    run(args, model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader)
