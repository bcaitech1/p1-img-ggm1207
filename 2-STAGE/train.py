import time
from argparse import Namespace

import torch
from sklearn.metrics import accuracy_score

from config import get_args
from prepare import load_sample
from database import execute_query
from slack import hook_fail_strategy
from networks import load_model_and_tokenizer
from utils import EarlyStopping, get_auto_save_path


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

        inputs = {
            "input_ids": batch["input_ids"].to(args.device),
            "attention_mask": batch["attention_mask"].to(args.device),
            "token_type_ids": batch["token_type_ids"].to(args.device),
        }

        labels = batch["labels"].to(args.device)

        preds = model(**inputs)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

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

            preds = model(**inputs)

            if "loss" in return_keys:
                loss = loss_fn(preds, labels)
                epoch_loss += loss.item()

            all_labels.extend(labels.detach().cpu().tolist())
            print(preds.argmax(-1))
            all_preds.extend(preds.detach().argmax(-1).cpu().tolist())

    if "loss" in return_keys:
        results["loss"] = epoch_loss / len(dataloader)

    if "acc" in return_keys:
        results["acc"] = accuracy_score(all_labels, all_preds)

    if "preds" in return_keys:
        results["preds"] = all_preds
    print(all_preds)
    return results


def run(args, model, loss_fn, optimizer, scheduler, train_dataloader, test_dataloader):
    """ train, evaluate for range(epochs), no hyperparameter search """
    args.save_path, _ = get_auto_save_path(args)
    early_stop = EarlyStopping(args, verbose=True)

    if isinstance(args, dict):
        args = Namespace(**args)

    for epoch in range(int(args.epochs)):
        start_time = time.time()

        train_loss = train(args, model, loss_fn, optimizer, scheduler, train_dataloader)
        results = evaluate(
            args, model, loss_fn, test_dataloader, return_keys=["loss", "acc"]
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        early_stop(train_loss, results["loss"], results["acc"], model)

        if early_stop.early_stop is True:
            break

        print()


def debug(args, strategy):
    """ just testing, is it works? """

    if isinstance(args, dict):
        args = Namespace(**args)

    args.batch_size = 32

    try:
        model, tokenizer = load_model_and_tokenizer(args)
        loss_fn = get_lossfn(args)

        inputs, labels = load_sample(args, tokenizer)
        preds = model(**inputs)
        _ = loss_fn(preds, labels)

        query = f"UPDATE STRATEGY SET STATUS = 'RUN' WHERE strategy='{strategy}'"
        execute_query(query)

    except Exception as e:
        print(e.with_traceback())
        query = f"UPDATE STRATEGY SET STATUS = 'PENDING' WHERE strategy='{strategy}'"
        execute_query(query)
        hook_fail_strategy(strategy, e.with_traceback())


if __name__ == "__main__":
    from losses import get_lossfn
    from prepare import load_dataloader
    from optimizers import get_optimizer, get_scheduler

    args = get_args()

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    train_dataloader, test_dataloader = load_dataloader(args, tokenizer)
    loss_fn = get_lossfn(args)
    optimizer = get_optimizer(args, model)
    #  scheduler = get_scheduler(args, optimizer)

    run(args, model, loss_fn, optimizer, None, train_dataloader, test_dataloader)
