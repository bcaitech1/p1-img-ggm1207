import os
from argparse import Namespace

import torch
import wandb
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.wandb import wandb_mixin

from config import get_args
from losses import FocalLoss
from prepare import load_dataloader
from train import train, evaluate, debug
from networks import load_model_and_tokenizer
from optimizers import get_optimizer, get_scheduler

from database import sample_strategy
from inference import if_best_score_auto_submit
from hook_slack import hook_simple_text
from utils import update_args, EarlyStopping


@wandb_mixin
def main(config, checkpoint_dir=None):
    step = 0
    args = Namespace(**config)

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    train_dataloader, test_dataloader = load_dataloader(args, tokenizer)
    loss_fn = FocalLoss(gamma=3)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)  # 안 쓰긴 하는데

    if checkpoint_dir is not None:  # Use For PBT
        print("I'm in checkpoint_dir!!")
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        step = checkpoint["step"]

    es_helper = EarlyStopping(args, verbose=True)  # Use For Ensemble

    while True:
        train_loss = train(args, model, loss_fn, optimizer, scheduler, train_dataloader)
        valid_loss, valid_acc = evaluate(args, model, loss_fn, test_dataloader)

        es_helper(train_loss, valid_loss, valid_acc, model)

        # wandb.log는 tune.report, tune.checkpoint_dir 보다 선행 되어야 한다.
        wandb.log(
            dict(valid_loss=valid_loss, valid_acc=valid_acc, train_loss=train_loss)
        )

        # 뭔지 모르겠지만 여기서 걍 끝남.
        tune.report(valid_loss=valid_loss, valid_acc=valid_acc, train_loss=train_loss)

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


def raytune(args):
    """ 하이퍼파라미터 설정하는 곳 """
    if isinstance(args, Namespace):
        args = vars(args)  # Namespace to dict

    while True:
        # status가 ready면 우선 Debug로 잘 돌아가는지 실험해보자.
        strategy, status, _, _ = sample_strategy()

        if status == "READY":
            debug(args, strategy)
            torch.cuda.empty_cache()  # Debug 이후에 할당된 메모리 해제
            continue

        # update hp_space, dataset_idx, wandb, save_path, base_name
        args = update_args(args, strategy)

        scheduler = PopulationBasedTraining(
            perturbation_interval=1,
            hyperparam_mutations={
                "learning_rate": tune.uniform(0.0001, 1),
                "weight_decay": tune.uniform(0.001, 0.05),
            },
        )

        hook_simple_text(f":pray: {args['base_name']} PBT 시작합니다!!")

        tune.run(
            main,
            name="pbt_bert_test",
            scheduler=scheduler,
            stop={"training_iteration": args["epochs"]},
            metric="valid_loss",
            mode="min",
            keep_checkpoints_num=3,
            num_samples=3,
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=args,
        )

        torch.cuda.empty_cache()

        hook_simple_text(f":joy: {args['base_name']} 학습 끝!!!")

        # if valid score is best, auto submission
        if_best_score_auto_submit(args["save_path"])

        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    raytune(args)
