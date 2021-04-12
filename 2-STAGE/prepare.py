import pickle
import os.path as p

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class RE_Dataset(Dataset):
    """ 어디서 사용되는 거지? """

    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset, label_type):
    """ 원하는 형태의 dataset으로 전처리 """
    label = []
    for i in dataset[8]:
        if i == "blind":
            label.append(100)
        else:
            label.append(label_type[i])

    out_dataset = pd.DataFrame(
        {"sentence": dataset[1], "entity_01": dataset[2], "entity_02": dataset[5]}
    )

    return out_dataset, label


def load_data(args, is_train=True):
    set_type = "train" if is_train else "test"

    label_path = p.join(args.data_dir, "label_type.pkl")
    file_path = p.join(args.data_dir, f"{set_type}.tsv")

    with open(label_path, "rb") as f:
        label_type = pickle.load(f)

    dataset = pd.read_csv(file_path, delimiter="\t", header=None)  # 왜 Tab으로 나눴을까?
    dataset, label = preprocessing_dataset(dataset, label_type)

    return dataset, label


def tokenized_dataset(dataset, tokenizer):
    concat_entity = []

    for e01, e02 in zip(dataset["entity_01"], dataset["entity_02"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )

    return tokenized_sentences


def load_dataset(args, is_train=True):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset, label = load_data(args, is_train=is_train)

    # word to token
    to_dataset = tokenized_dataset(dataset, tokenizer)

    # hmm... return (dict type)
    re_train_dataset = RE_Dataset(to_dataset, label)
    return re_train_dataset, tokenizer
