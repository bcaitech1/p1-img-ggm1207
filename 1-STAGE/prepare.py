import os
import time
import numpy as np
from glob import glob
from PIL import Image

import cv2
import pandas as pd
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from config import get_args


def get_classes(key):
    """ predict하기 위해서는 순서가 중요하다. """
    if key == "mask":
        return ["wear", "incorrect", "not wear"]
    if key == "age":
        return ["age < 30", "30 <= age < 60", "60 <= age"]
    if key == "gender":
        return ["male", "female"]
    raise KeyError("key must be in ['mask', 'age', 'gender']")


def get_transforms(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform


def get_album_transforms(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trans_fns = [
        A.CoarseDropout(max_width=50, max_height=50, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.ColorJitter(p=0.5),
        A.Cutout(p=0.5, max_h_size=50, max_w_size=50),
        A.FancyPCA(alpha=0.5, p=0.5),
        A.GridDistortion(p=0.5),
        A.GridDropout(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGridShuffle(p=0.5),
        A.ToGray(p=1),  # 12
    ]

    # (결국에는) Trasnsform을 만들어서 사용하는 것이 좋다.

    trans_fn = trans_fns[args.temp_aug_index]
    train_transform = A.Compose(
        [
            A.Resize(args.image_size, args.image_size),
            trans_fn,  # keep uint8
            A.Normalize(mean, std),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(args.image_size, args.image_size),
            A.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


class MaskDataSet(Dataset):
    def __init__(self, args, is_train=True, transform=None):
        csv_file = os.path.join(args.data_dir, "train.csv")
        self.datas = pd.read_csv(csv_file)
        self.images, self.labels = self._load_image_files_path(args, is_train)
        self.label_idx = ["gender", "age", "mask"].index(args.train_key)
        self.args = args

        if args.test:
            self.images, self.labels = self.images[:100], self.labels[:100]

        self.transform = transform

    def __getitem__(self, idx):

        img = Image.open(self.images[idx])
        img = np.array(img)  # time: 16.8463

        if self.transform:
            img = self.transform(image=img)["image"]

        # Share Memory
        img = np.transpose(img, axes=(2, 0, 1))  # (w, h, c) +> (c, w, h)

        return img, self.labels[idx][self.label_idx]

    def __len__(self):
        return len(self.images)

    def _load_image_files_path(self, args, is_train):
        split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.valid_size,
            random_state=0,  # 이 SEED값은 안 바꾸는 것이 좋다.
        )

        split_key = "age" if args.train_key == "age" else "gender"

        for train_index, valid_index in split.split(self.datas, self.datas[split_key]):
            train_dataset = self.datas.loc[train_index]
            valid_dataset = self.datas.loc[valid_index]

        dataset = train_dataset if is_train else valid_dataset
        gender_classes = get_classes("gender")

        images = []
        labels = []

        for dir_name in dataset["path"]:
            dir_path = os.path.join(args.data_dir, "images", dir_name)

            image_id, gender_lbl, _, age_lbl = dir_name.split("_")

            gender_class = gender_classes.index(gender_lbl)

            # ["age < 30", "30 <= age < 60", "60 <= age"]
            age_lbl = int(age_lbl)

            if age_lbl < 30:
                age_class = 0
            elif age_lbl >= 60:
                age_class = 2
            else:
                age_class = 1

            for jpg_filepath in glob(dir_path + "/*"):
                jpg_basename = os.path.basename(jpg_filepath)

                if "normal" in jpg_basename:
                    mask_class = 2
                elif "incorrect" in jpg_basename:
                    mask_class = 1
                else:
                    mask_class = 0

                images.append(jpg_filepath)
                labels.append((gender_class, age_class, mask_class))

        return images, labels


def get_dataloader(args):
    train_transform, test_transform = get_album_transforms(args)

    train_dataset = MaskDataSet(args, is_train=True, transform=train_transform)
    valid_dataset = MaskDataSet(args, is_train=False, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    args = get_args()

    train_dataloader, valid_dataloader = get_dataloader(args)
    s_time = time.time()
    for idx, (images, labels) in enumerate(train_dataloader):
        pass

    for idx, (images, labels) in enumerate(train_dataloader):
        pass
    print(time.time() - s_time)