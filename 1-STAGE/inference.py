""" 모델을 Load한 후 전체 데이터에 대해서 다시 학습 한 후 Inference 실행 """

import os
import json
from PIL import Image

import torch
import pandas as pd
import torch.nn as nn

import train
from config import get_args
from predict import load_models
from metrics import calculate_18class
from prepare import get_dataloader, get_transforms


def load_args(model_path):
    arg_path = model_path[:-2] + "args"
    with open(arg_path, "r") as f:
        args = json.loads(f.readline())
    return args


def retrain(args):
    """ args.*_model """
    
    print(args)
    model_pathes = [args.age_model, args.gender_model, args.mask_model]

    for model_path in model_pathes:
        train_args = load_args(model_path)

        train_args["valid_size"] = 0
        train_args["model_path"] = "/opt/ml/inference_weights/"
        train_args["epochs"] = 100
        train_args["eval"] = False

        train.main(train_args)

def main(args):

    retrain(args)

    #  eval_dir = "/opt/ml/input/data/eval/"
    #  eval_df = pd.read_csv(os.path.join(eval_dir, "info.csv"))
    #
    #  for idx, image_base_path in enumerate(eval_df["ImageID"]):
    #      image_full_path = os.path.join(eval_dir, "images", image_base_path)
    #      image = Image.open(image_full_path)
    #      image = transform(image).unsqueeze(0).cuda()
    #
    #      age_label = age_model(image)
    #      gender_label = gender_model(image)
    #      mask_label = mask_model(image)
    #
    #      age_class = torch.argmax(age_label, dim=1)
    #      gender_class = torch.argmax(gender_label, dim=1)
    #      mask_class = torch.argmax(mask_label, dim=1)
    #
    #      res = eval_class(mask_class.item(), gender_class.item(), age_class.item())
    #      eval_df.iloc[idx, 1] = res
    #
    #      print(idx, end="\r")
    #
    #  sub_path = "/opt/ml/P-Stage/1-STAGE/submissions"
    #  sub_path = os.path.join(sub_path, f"{args.inf_filename}-submission.csv")
    #  eval_df.to_csv(sub_path, index=False)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
