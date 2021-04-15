import os
import time
import math
import pickle
import random
import pandas as pd
from argparse import Namespace

import ray
import torch
import wandb
import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.wandb import wandb_mixin

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
from hook_slack import hook_fail_strategy, hook_fail_ray, hook_simple_text
from inference import check_last_valid_score
from losses import FocalLoss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(args, model, loss_fn, optimizer, scheduler, dataloader):
    model.train()
    epoch_loss = 0.0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"].to(args.device),
            "attention_mask": batch["attention_mask"].to(args.device),
            #  "labels": batch["label_ids"].to(args.device),
        }

        labels = batch["label_ids"].to(args.device)

        outputs = model(**inputs)
        loss = loss_fn(outputs, labels)

        #  loss = outputs[0]

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(args, model, loss_fn, dataloader):
    model.eval()
    epoch_loss = 0.0

    total_len = 0
    correct_len = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                #  "labels": batch["label_ids"].to(args.device),
            }

            labels = batch["label_ids"].to(args.device)

            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)

            correct_len += torch.sum(labels.squeeze() == outputs.argmax(-1)).item()

            total_len += outputs.size(0)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader), correct_len / total_len


def run(args, model, loss_fn, optimizer, scheduler, train_dataloader, test_dataloader):
    #  es_helper = EarlyStopping(args, verbose=True)  # logging, save

    for epoch in range(int(args.epochs)):
        start_time = time.time()

        train_loss = train(args, model, loss_fn, optimizer, scheduler, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, loss_fn, test_dataloader)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        #  es_helper(train_loss, valid_loss, valid_acc, model)
        #
        #  if es_helper.early_stop:
        #      break


# batch_size, learning rate, momentum, beta1, beta2, weight decay,
# learning rate scheduler
# optimizer
# accumulation steps -> batch_size랑 관련 있는 거였어..


@wandb_mixin
def main(config, checkpoint_dir=None):
    step = 0
    args = Namespace(**config)

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    train_dataloader, test_dataloader = load_dataloader(args, tokenizer)
    loss_fn = FocalLoss(gamma=3)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)  # 안 쓰긴 하는데

    #  run(args, model, loss_fn, optimizer, scheduler, train_dataloader, test_dataloader)

    if checkpoint_dir is not None:  # Use For PBT
        print("I'm in checkpoint_dir!!")
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        step = checkpoint["step"]

    es_helper = EarlyStopping(args, verbose=True)  # Use For Ensemble

    while True:
        start_time = time.time()

        train_loss = train(args, model, loss_fn, optimizer, scheduler, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, loss_fn, test_dataloader)

        es_helper(train_loss, valid_loss, valid_acc, model)

        # wandb.log는 tune.report, tune.checkpoint_dir 보다 선행 되어야 한다.
        wandb.log(
            dict(valid_loss=valid_loss, valid_acc=valid_acc, train_loss=train_loss)
        )

        # 뭔지 모르겠지만 여기서 걍 끝남.
        tune.report(valid_loss=valid_loss, valid_acc=valid_acc, train_loss=train_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

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


def debug(args, strategy):
    args = Namespace(**args)

    try:
        model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
        train_dataloader, test_dataloader = load_dataloader(args, tokenizer)

        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)

        loss_fn = FocalLoss(gamma=3)

        run(
            args,
            model,
            loss_fn,
            optimizer,
            scheduler,
            train_dataloader,
            test_dataloader,
        )

        query = f"UPDATE STRATEGY SET STATUS = 'RUN' WHERE strategy='{strategy}'"
        execute_query(query)

    except Exception as e:
        query = f"UPDATE STRATEGY SET STATUS = 'PENDING' WHERE strategy='{strategy}'"
        execute_query(query)
        hook_fail_strategy(strategy, e)


def raytune(args):
    """ 하이퍼파라미터 설정하는 곳 """
    args = vars(args)  # update_args(args)

    while True:
        # status가 ready면 우선 Debug로 잘 돌아가는지 실험해보자.
        strategy, status, cnt, v_avg_score = sample_strategy()

        if status == "READY":
            origin_args = vars(get_args())
            args.update(
                {
                    k: v
                    for k, v in origin_args.items()
                    if k in {"learning_rate", "batch_size", "weight_decay"}
                }
            )

            args["debug"] = True
            args["epochs"] = 1
            debug(args, strategy)
            torch.cuda.empty_cache()  # Debug 이후에 할당된 메모리 해제
            continue

        # Debug를 통과하면 메모리 할당 문제도 해결 됨.

        args["strategy"] = strategy
        args = update_args(args)
        args["dataset_idx"] = random.randint(0, 4)  # 0 ~ 4

        save_path = get_auto_save_path(Namespace(**args))
        base_name = os.path.basename(save_path)[:-4]

        args["wandb"] = {
            "project": "p-stage-2",
            "api_key": "b9adc17bf9dff02b1aa29666268b7ab9ccaf2e56",
            "name": base_name,
        }

        args["save_path"] = save_path

        #  wandb.run.name = base_name

        scheduler = PopulationBasedTraining(
            perturbation_interval=5,
            hyperparam_mutations={
                "learning_rate": lambda: np.random.uniform(0.0001, 1),
                "weight_decay": lambda: np.random.uniform(0.001, 0.05),
            },
        )

        hook_simple_text(f":pray: {base_name} PBT 시작합니다!!")

        tune.run(
            main,
            name="pbt_bert_test",
            scheduler=scheduler,
            stop={"training_iteration": args["epochs"]},
            metric="valid_loss",
            mode="min",
            keep_checkpoints_num=5,
            num_samples=2,
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=args,
        )

        torch.cuda.empty_cache()

        hook_simple_text(f":joy: {base_name} 학습 끝!!!")

        check_last_valid_score(args, save_path)


if __name__ == "__main__":
    ray.init()
    args = get_args()

    raytune(args)
    #  try:
    #  except Exception as e:
    #      print(e)
    #      hook_fail_ray()
