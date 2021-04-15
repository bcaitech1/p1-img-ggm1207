import os
import glob
import json
import os.path as p
from argparse import Namespace

import torch
import numpy as np

from hp import strat


def custom_model_save(model, save_path):
    torch.save(model.state_dict(), save_path)


def get_auto_save_path(args):
    prefix = "_".join([args.strategy, args.ms_name])
    prefix = p.join(args.weight_dir, prefix)

    s_cnt = len(glob.glob(prefix + "*"))
    save_path = f"{prefix}_{s_cnt:03}.pth"

    assert not p.exists(save_path), f"{save_path} already exists"

    return save_path


def update_args(args):
    if isinstance(args, Namespace):
        args = vars(args)

    """ return dict type """
    arj = strat[args["strategy"]]
    args.update(arj)
    return args


class EarlyStopping:
    def __init__(self, args, verbose=True):
        self.args = args
        self.counter = 0
        self.verbose = verbose
        self.dalta = args.delta  # Delta값 기준으로 성능 개선을 판단한다.
        self.early_stop = False
        self.patience = args.patience
        self.best_valid_loss = float("inf")

        self.model_save_path = args.save_path

    def __call__(self, train_loss, valid_loss, valid_acc, model):

        if self.verbose:
            print(f"\tTrain Loss: {train_loss:.4f}")
            print(f"\tValidation Loss: {valid_loss:.4f}")
            print(f"\tValidation Acc: {valid_acc:.4f}")

        if valid_loss < self.best_valid_loss - self.dalta:
            self.save_checkpoint(valid_loss, model)
            self.best_valid_loss = valid_loss

        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, valid_loss, model):

        if self.verbose:
            print(
                f"Validation loss decreased ({self.best_valid_loss:.4f} --> {valid_loss:.4f}).  Saving model ..."
            )

        custom_model_save(model, self.model_save_path)
