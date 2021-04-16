import pickle
import random
import os.path as p

import torch
from torch.utils.data import Dataset, DataLoader


class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        return item

    def __len__(self):
        return len(self.tokenized_dataset["labels"])


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

    re_tt_dataset = RE_Dataset(tt_dataset)
    re_tv_dataset = RE_Dataset(tv_dataset)

    train_dataloader = DataLoader(
        re_tt_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # Load Faster, If Use GPU
        num_workers=1,  # 이미 Memory에 다 올렸는데, 굳이 개수가 많을 필요가 있을까?
        drop_last=True,
    )

    test_dataloader = DataLoader(
        re_tv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    return train_dataloader, test_dataloader


def tokenized_dataset(args, dataset, tokenizer):
    # dataset keywords: "words", "e1", "e2", "labels"

    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_label_ids = []

    special_tokens_count = 1  # 사실 4개임

    for idx, (e01, e02, words, label) in enumerate(
        zip(dataset["e1"], dataset["e2"], dataset["words"], dataset["labels"])
    ):
        # Using Tokenizer Encode, 저절로 [CLS] , [SEP] 는 붙음

        tokens = [e01, e02] + words
        all_label_ids.append(label)

        tokens = tokenizer.encode(" ".join(tokens))

        if len(tokens) > args.max_seq_length - special_tokens_count:
            tokens = tokens[: (args.max_seq_length - special_tokens_count)]

            tokens.append(
                tokenizer.convert_tokens_to_ids(
                    tokenizer.special_tokens_map["unk_token"]
                )
            )

        token_type_ids = [0] * len(tokens)

        input_ids = tokens
        attention_mask = [1] * len(input_ids)

        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length

        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length  # 질문 문제가 아니라서..

        assert len(input_ids) == args.max_seq_length
        assert len(attention_mask) == args.max_seq_length
        assert len(token_type_ids) == args.max_seq_length

        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)

        if args.debug is True and idx == 100:
            break

    tokenized_sentences = {
        "input_ids": torch.tensor(all_input_ids),
        "token_type_ids": torch.tensor(all_token_type_ids),
        "attention_mask": torch.tensor(all_attention_mask),
        "labels": torch.tensor(all_label_ids),
    }

    return tokenized_sentences


#  def preprocessing_dataset(dataset, label_type):
#      """ 원하는 형태의 dataset으로 전처리 """
#      label = []
#      for i in dataset[8]:
#          if i == "blind":
#              label.append(100)
#          else:
#              label.append(label_type[i])
#
#      out_dataset = pd.DataFrame(
#          {"sentence": dataset[1], "entity_01": dataset[2], "entity_02": dataset[5]}
#      )
#
#      return out_dataset, label


#  def tokenized_dataset(args, dataset, tokenizer):
#      # dataset keywords: "words", "e1", "e2", "labels"
#      concat_entity = []
#
#      for e01, e02 in zip(dataset["e1"], dataset["e2"]):
#          temp = e01 + "[SEP]" + e02
#          concat_entity.append(temp)
#
#      tokenized_sentences = tokenizer(
#          concat_entity,
#          dataset["words"],
#          return_tensors="pt",
#          padding=True,
#          truncation=True,
#          add_special_tokens=True,
#          max_length=args.token_max_length,
#      )
#
#      return tokenized_sentences


#  def load_dataset(args, tokenizer, is_train=True):
#      tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#      dataset, label = load_data(args, is_train=is_train)
#
#      # word to token
#      to_dataset = tokenized_dataset(dataset, tokenizer)
#
#      # hmm... return (dict type)
#      re_train_dataset = RE_Dataset(to_dataset, label)
#      return re_train_dataset, tokenizer

if __name__ == "__main__":
    from config import get_args

    args = get_args()
    dataset, valid_dataset = pick_one_dataset(args, is_train=True)

    for idx, (e01, e02, words, label) in enumerate(
        zip(dataset["e1"], dataset["e2"], dataset["words"], dataset["labels"])
    ):
        print(e01, e02, words, label)
