""" 모델을 Load한 후 전체 데이터에 대해서 다시 학습 한 후 Inference 실행 """

import os
from PIL import Image

import torch
import pandas as pd
import torch.nn as nn

from train import evaluate
from config import get_args
from prepare import get_dataloader, get_transforms
from metrics import change_2d_to_1d, change_age_to_cat, cal_metrics, cal_accuracy


def eval_class(mi, gi, ai):
    return 6 * mi + 3 * gi + ai


def main(args):
    print(args)

    try:
        age_model = torch.load(args.age_model)
        gender_model = torch.load(args.gender_model)
        mask_model = torch.load(args.mask_model)
    except Exception as e:
        print(e)
        raise "Failed Model load"

    age_model.eval()
    gender_model.eval()
    mask_model.eval()

    transform = get_transforms(args)

    eval_dir = "/opt/ml/input/data/eval/"
    eval_df = pd.read_csv(os.path.join(eval_dir, "info.csv"))


    for idx, image_base_path in enumerate(eval_df["ImageID"]):
        image_full_path = os.path.join(eval_dir, "images", image_base_path)
        image = Image.open(image_full_path)
        image = transform(image).unsqueeze(0).cuda()

        age_label = age_model(image)
        gender_label = gender_model(image)
        mask_label = mask_model(image)

        age_class = torch.argmax(age_label, dim=1)
        gender_class = torch.argmax(gender_label, dim=1)
        mask_class = torch.argmax(mask_label, dim=1)

        res = eval_class(mask_class.item(), gender_class.item(), age_class.item())
        eval_df.iloc[idx, 1] = res

        print(idx, end='\r')

    sub_path = "/opt/ml/P-Stage/1-STAGE/submissions"
    sub_path = os.path.join(sub_path, f"{args.filename}-submission.csv")
    eval_df.to_csv(sub_path, index=False)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
