import os
import time
import math
import pickle
import random
import pandas as pd
from argparse import Namespace

import ray
import wandb
import torch
import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from transformers import (
    Trainer,
    BertConfig,
    TrainingArguments,
    BertForSequenceClassification,
)

from config import get_args
from prepare import load_dataloader
from networks import load_model_and_tokenizer
from optimizers import get_optimizer, get_scheduler
from database import sample_strategy, execute_query
from utils import custom_model_save, get_auto_save_path, update_args, EarlyStopping


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(args, model, optimizer, scheduler, dataloader):
    model.train()
    epoch_loss = 0.0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"].to(args.device),
            "attention_mask": batch["attention_mask"].to(args.device),
            "labels": batch["label_ids"].to(args.device),
        }

        outputs = model(**inputs)
        loss = outputs[0]

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, dataloader):
    model.eval()
    epoch_loss = 0.0

    total_len = 0
    correct_len = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "labels": batch["label_ids"].to(args.device),
            }

            outputs = model(**inputs)
            loss, preds = outputs[:2]

            correct_len += torch.sum(
                inputs["labels"].squeeze() == preds.argmax(-1)
            ).item()

            total_len += preds.size(0)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader), correct_len / total_len


def run(args, model, optimizer, scheduler, train_dataloader, test_dataloader):
    es_helper = EarlyStopping(args, verbose=True)  # logging, save

    for epoch in range(int(args.epochs)):
        start_time = time.time()

        train_loss = train(args, model, optimizer, scheduler, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, test_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        es_helper(train_loss, valid_loss, valid_acc, model)

        if es_helper.early_stop:
            break


# batch_size, learning rate, momentum, beta1, beta2, weight decay,
# learning rate scheduler
# optimizer
# accumulation steps -> batch_size랑 관련 있는 거였어..


def main(config, checkpoint_dir=None):
    step = 0
    args = Namespace(**config)
    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)

    train_dataloader, test_dataloader = load_dataloader(args, tokenizer)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)  # 안 쓰긴 하는데
    #  run(args, model, optimizer, scheduler, train_dataloader, test_dataloader)

    if checkpoint_dir is not None:  # Use For PBT
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        step = checkpoint["step"]

    es_helper = EarlyStopping(args, verbose=True)  # Use For Ensemble

    while True:
        start_time = time.time()

        train_loss = train(args, model, optimizer, scheduler, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, test_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        es_helper(train_loss, valid_loss, valid_acc, model)

        #  if es_helper.early_stop:
        #      break

        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step,
                },
                path,
            )

        tune.report(valid_loss=valid_loss, valid_acc=valid_acc, train_loss=train_loss)


def debug(args, strategy):
    args = Namespace(**args)

    try:
        model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
        train_dataloader, test_dataloader = load_dataloader(args, tokenizer)

        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)

        run(args, model, optimizer, scheduler, train_dataloader, test_dataloader)

        query = f"UPDATE STRATEGY SET STATUS = 'RUN' WHERE strategy='{strategy}'"
        execute_query(query)

    except Exception as e:
        print(e)
        query = f"UPDATE STRATEGY SET STATUS = 'PENDING' WHERE strategy='{strategy}'"
        execute_query(query)


def raytune(args):
    """ 하이퍼파라미터 설정하는 곳 """
    args = vars(args)  # update_args(args)
    #  ray.init(webui_host=’127.0.0.1’)

    # 우선 strategy마다 같은 전략 사용.

    while True:
        # status가 ready면 우선 Debug로 잘 돌아가는지 실험해보자.
        strategy, status, cnt, v_avg_score = sample_strategy()

        if status == "READY":
            args = vars(get_args())
            args["debug"] = True
            debug(args, strategy)
            torch.cuda.empty_cache()  # Debug 이후에 할당된 메모리 해제
            continue

        # Debug를 통과하면 메모리 할당 문제도 해결 됨.

        args["strategy"] = strategy
        args = update_args(args)
        args["dataset_idx"] = random.randint(0, 4)  # 0 ~ 4

        scheduler = PopulationBasedTraining(
            perturbation_interval=5,
            hyperparam_mutations={
                "learning_rate": lambda: np.random.uniform(0.0001, 1),
                "weight_decay": lambda: np.random.uniform(0.001, 0.05),
            },
        )

        tune.run(
            main,
            name="pbt_bert_test",
            scheduler=scheduler,
            stop={"training_iteration": args["epochs"]},
            metric="valid_loss",
            mode="min",
            keep_checkpoints_num=5,
            num_samples=2,
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=args,
        )

        query = f"UPDATE STRATEGY SET cnt = {cnt+1} WHERE strategy = '{strategy}'"
        execute_query(query)


if __name__ == "__main__":
    ray.init()
    args = get_args()

    raytune(args)
