import os
import pickle
import pandas as pd

import wandb
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    Trainer,
    BertConfig,
    TrainingArguments,
    BertForSequenceClassification,
)

from config import get_args
from prepare import load_dataset

#  from src import (
#      CONFIG_CLASSES,
#      TOKENIZER_CLASSES,
#      MODEL_FOR_TOKEN_CLASSIFICATION,
#      init_logger,
#      set_seed,
#      compute_metrics,
#      show_ner_report,
#  )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main(args):
    train_dataset, label = load_dataset(args, is_train=True)
    valid_dataset, label = load_dataset(args, is_train=False)

    model_config = BertConfig.from_pretrained(args.model_name_or_path)
    model_config.num_labels = args.num_labels

    model = BertForSequenceClassification(model_config)
    model.to(args.device)

    ta_args = [
        "output_dir",
        "save_total_limit",
        "save_steps",
        "num_train_epochs",
        "learning_rate",
        "per_device_eval_batch_size",
        "per_device_train_batch_size",
        "warmup_steps",
        "weights_decay",
        "logging_dir",
        "logging_steps",
        "evaluation_strategy",
        "eval_steps",
    ]

    ta = {k: v for k, v in vars(args).items() if k in ta_args}

    t_args = TrainingArguments(**ta)  # vars : NameSpace to Dict

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(args)
