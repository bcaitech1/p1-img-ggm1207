import pickle
import random
import os.path as p

import torch
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import Dataset, DataLoader


class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def _get_label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.labels)


def pick_one_dataset(args, is_train=True):
    data_path = p.join(args.data_dir, args.data_kind)

    if is_train:
        idx = args.dataset_idx

        train_path = p.join(data_path, f"train-{idx}.pkl")
        valid_path = p.join(data_path, f"valid-{idx}.pkl")

        with open(train_path, "rb") as f:
            train_dataset = pickle.load(f)

        with open(valid_path, "rb") as f:
            valid_dataset = pickle.load(f)

        return train_dataset, valid_dataset
    else:
        test_path = p.join(data_path, "test.pkl")

        with open(test_path, "rb") as f:
            test_dataset = pickle.load(f)

        return test_dataset


def load_test_dataloader(args, tokenizer):
    test_dataset = pick_one_dataset(args, is_train=False)
    test_dataset = tokenized_dataset(args, test_dataset, tokenizer)
    test_dataset = RE_Dataset(test_dataset)

    test_dataloader = DataLoader(
        test_dataset, batch_size=32, pin_memory=True, num_workers=1, shuffle=False
    )

    return test_dataloader


def load_sample(args, tokenizer):
    _, valid_dataset = pick_one_dataset(args, is_train=True)

    tv_dataset = tokenized_dataset(args, valid_dataset, tokenizer)
    re_tv_dataset = RE_Dataset(tv_dataset)

    idx = random.randint(0, len(re_tv_dataset))
    batch = re_tv_dataset[idx]

    inputs = {
        "input_ids": batch["input_ids"].to(args.device).unsqueeze(0),
        "attention_mask": batch["attention_mask"].to(args.device).unsqueeze(0),
        "token_type_ids": batch["token_type_ids"].to(args.device).unsqueeze(0),
    }

    labels = batch["labels"].to(args.device)

    return inputs, labels


def load_dataloader(args, tokenizer):
    train_dataset, valid_dataset = pick_one_dataset(args, is_train=True)

    tt_dataset = tokenized_dataset(args, train_dataset, tokenizer)
    tv_dataset = tokenized_dataset(args, valid_dataset, tokenizer)

    re_tt_dataset = RE_Dataset(tt_dataset, train_dataset["labels"])
    re_tv_dataset = RE_Dataset(tv_dataset, valid_dataset["labels"])

    def callback_get_label(dataset, idx):
        return dataset._get_label(idx)

    train_dataloader = DataLoader(
        re_tt_dataset,
        batch_size=args.batch_size,
        #  shuffle=True,
        pin_memory=True,  # Load Faster, If Use GPU
        num_workers=1,  # 이미 Memory에 다 올렸는데, 굳이 개수가 많을 필요가 있을까?
        sampler=ImbalancedDatasetSampler(
            re_tt_dataset, callback_get_label=callback_get_label
        ),
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        re_tv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    return train_dataloader, valid_dataloader


def tokenized_dataset(args, dataset, tokenizer):
    concat_entity = []

    for idx, (e01, e02) in enumerate(zip(dataset["e1"], dataset["e2"])):
        concat_entity.append(e01 + "[SEP]" + e02)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["words"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences


if __name__ == "__main__":
    from config import get_args

    args = get_args()
    dataset, valid_dataset = pick_one_dataset(args, is_train=True)

    for idx, (e01, e02, words, label) in enumerate(
        zip(dataset["e1"], dataset["e2"], dataset["words"], dataset["labels"])
    ):
        print(e01, e02, words, label)
        break
