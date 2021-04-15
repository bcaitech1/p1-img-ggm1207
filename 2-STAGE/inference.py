import os
import re
import json
import requests
from argparse import Namespace

import torch
import numpy as np
import pandas as pd

from utils import get_auto_save_path
from prepare import load_dataloader, load_test_dataloader
from networks import load_model_and_tokenizer
from losses import FocalLoss
from database import (
    execute_query,
    get_scores_of_strategy,
    update_strategy_statistics,
    insert_model_scores,
)


def auto_submit(user_key, description, file_path):
    url = (
        f"http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/4/presigned_url/?description={description}"
        + "&hyperparameters={%22training%22:{},%22inference%22:{}}"
    )

    headers = {"Authorization": user_key}
    res = requests.get(url, headers=headers)
    data = json.loads(res.text)
    submit_url = data["url"]

    body = {
        "key": "app/Competitions/000004/Users/{}/Submissions/{}/output.csv".format(
            str(data["submission"]["user"]).zfill(8),
            str(data["submission"]["local_id"]).zfill(4),
        ),
        "x-amz-algorithm": data["fields"]["x-amz-algorithm"],
        "x-amz-credential": data["fields"]["x-amz-credential"],
        "x-amz-date": data["fields"]["x-amz-date"],
        "policy": data["fields"]["policy"],
        "x-amz-signature": data["fields"]["x-amz-signature"],
    }

    requests.post(url=submit_url, data=body, files={"file": open(file_path, "rb")})


def check_last_valid_score(args, save_path):
    if isinstance(args, dict):
        args = Namespace(**args)

    args.batch_size = 32

    model, tokenizer = load_model_and_tokenizer(args)  # to(args.device)
    _, valid_dataloader = load_dataloader(args, tokenizer)

    model.load_state_dict(torch.load(save_path))

    loss_fn = FocalLoss(gamma=3)

    model.eval()
    epoch_loss = 0.0

    total_len = 0
    correct_len = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
                #  "labels": batch["label_ids"].to(args.device),
            }

            labels = batch["label_ids"].to(args.device)

            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)

            correct_len += torch.sum(labels.squeeze() == outputs.argmax(-1)).item()

            total_len += outputs.size(0)
            epoch_loss += loss.item()

    valid_avg_loss = epoch_loss / len(valid_dataloader)
    valid_acc = correct_len / total_len

    s_cnt = re.findall("[\d]{3}", save_path)[-1]

    # UPDATE Strategy
    is_submit = update_strategy_statistics(args, valid_acc)
    # INSERT MODEL, Model별 성능 기록
    insert_model_scores(args, s_cnt, valid_acc)

    if (is_submit is True) and args.auto_sub:
        submission_inference(args, model, tokenizer, save_path)


def submission_inference(args, model, tokenizer, save_path):
    """ save_path: .../st01_kobert_002.pth"""
    model.eval()

    base_name = os.path.basename(save_path)[:-4]
    save_path = os.path.join(args.submit_dir, base_name) + ".csv"

    test_dataloader = load_test_dataloader(args, tokenizer)
    output_pred = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "token_type_ids": batch["token_type_ids"].to(args.device),
            }

            logits = model(**inputs)
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.extend(result)

    preds = np.array(output_pred).reshape(-1)
    output = pd.DataFrame(preds, columns=["pred"])
    output.to_csv(save_path, index=False)

    user_key = "Bearer 5c12695179ea1f0a97aec9ce2be8da028755f095"
    auto_submit(user_key, base_name, save_path)


if __name__ == "__main__":
    from config import get_args

    args = get_args()
    check_last_valid_score(
        args,
        "/home/j-gunmo/desktop/00.my-project/17.P-Stage-T1003/2-STAGE/weights/st01_kobert_002.pth",
    )

    #  auto_submit(user_key, "description_test", "./submission.csv")
