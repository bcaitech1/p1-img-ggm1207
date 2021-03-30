import os
from PIL import Image

import wandb
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
    wandb.init(preject="p-stage-1", reinit=True)
    wandb.config.update(args)

    mse_loss = nn.MSELoss()
    cro_loss = nn.CrossEntropyLoss()

    try:
        age_model = torch.load(args.age_model)
        gender_model = torch.load(args.gender_model)
        mask_model = torch.load(args.mask_model)
    except E as e:
        print(e)
        raise "Failed Model load"

    _, valid_dataloader = get_dataloader(args)
    transform = get_transforms(args)

    eval_dir = "/opt/ml/input/data/eval/"
    eval_df = pd.read_csv(os.path.join(eval_dir, "info.csv"))

    summary_df = pd.DataFrame(columns=["f1_score", "accuracy"])

    args.train_key = "age"
    _, label_list, output_list = evaluate(args, age_model, mse_loss, valid_dataloader)
    f1_sco = cal_metrics(output_list, label_list)
    acc = cal_accuracy(output_list, label_list)
    summary_df.loc["age"] = [f1_sco, acc]

    args.train_key = "gender"
    _, label_list, output_list = evaluate(
        args, gender_model, cro_loss, valid_dataloader
    )
    f1_sco = cal_metrics(output_list, label_list)
    acc = cal_accuracy(output_list, label_list)
    summary_df.loc["gender"] = [f1_sco, acc]

    args.train_key = "mask"
    _, label_list, output_list = evaluate(args, mask_model, cro_loss, valid_dataloader)
    f1_sco = cal_metrics(output_list, label_list)
    acc = cal_accuracy(output_list, label_list)
    summary_df.loc["mask"] = [f1_sco, acc]

    table = wandb.Table(dataframe=summary_df)

    for idx, image_base_path in enumerate(eval_df["ImageID"]):
        image_full_path = os.path.join(eval_dir, "images", image_base_path)
        image = Image.open(image_full_path)
        image = transform(image).unsqueeze(0).cuda()

        age_label = age_model(image)
        gender_label = gender_model(image)
        mask_label = mask_model(image)

        age_class = change_age_to_cat(age_label[0])
        gender_class = torch.argmax(gender_label, dim=1)
        mask_class = torch.argmax(mask_label, dim=1)

        res = eval_class(mask_class.item(), gender_class.item(), age_class.item())
        eval_df.iloc[idx, 1] = res

    wandb.log({"Result": table})

    sub_path = "/opt/ml/P-Stage/1-STAGE/submissions"
    sub_path = os.path.join(sub_path, f"{wandb.run.name}.csv")
    eval_df.to_csv(sub_path)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
