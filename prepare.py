import os
import time
import numpy as np
from glob import glob
from PIL import Image
from functools import partial

import cv2
import pandas as pd
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from sklearn.model_selection import StratifiedShuffleSplit

from config import get_args
from autoaugment import ImageNetPolicy


def get_num_classes(args):
    if args.train_key == "mask":
        return 3
    if args.train_key == "age":
        return 3
    if args.train_key == "gender":
        return 2
    if args.train_key == "age-coral":  # 18 ~ 60
        return 60 - 18 + 1


def get_classes(args):
    """ predict하기 위해서는 순서가 중요하다. """
    if args.train_key == "mask":
        return ["wear", "incorrect", "not wear"]
    if args.train_key == "age":
        return ["age < 30", "30 <= age < 60", "60 <= age"]
    if args.train_key == "gender":
        return ["male", "female"]
    if args.train_key == "age-coral":
        return [str(i) for i in range(0, 43)]
    raise KeyError("key must be in ['mask', 'age', 'gender']")


def get_transforms(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, test_transform


def get_album_transforms(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    w = h = args.image_size // 3

    mask = np.zeros((args.image_size, args.image_size)).astype(np.uint8)
    mask[: h + h, :] = np.ones_like(mask[: h + h, :]).astype(np.uint8)

    #  trans_fns = [
    #      A.MaskDropout(p=0.5),  # 0
    #      A.ColorJitter(p=0.5),  # 1
    #      A.FancyPCA(alpha=0.5, p=0.5),
    #      A.GridDistortion(p=0.5),
    #      A.GridDropout(p=0.5),  # 4
    #      A.RandomBrightnessContrast(p=0.5),
    #      A.RandomGridShuffle(p=0.5),  # 6
    #      A.RandomGridShuffle(p=0.5, grid=(6, 6)),
    #      A.InvertImg(p=0.5),
    #  ]

    aug_keys = args.aug_keys.split(",")
    prob = 1 / (len(aug_keys) + 1)

    trans_fns = {
        "MD": A.MaskDropout(p=prob),
        "CJ": A.ColorJitter(p=prob),
        "FancyPCA": A.FancyPCA(alpha=prob, p=prob),
        "GridDist": A.GridDistortion(p=prob),
        "GridDrop": A.GridDropout(p=prob),
        "RandomBC": A.RandomBrightnessContrast(p=prob),
        "RGS_33": A.RandomGridShuffle(p=prob),
        "RGS_66": A.RandomGridShuffle(p=prob, grid=(6, 6)),
        "INV": A.InvertImg(p=prob),
        "CLAHE": A.CLAHE(p=prob),
    }

    # (결국에는) Trasnsform을 만들어서 사용하는 것이 좋다.

    new_trans_fns = [trans_fns[aug_key] for aug_key in aug_keys]

    # 이미지 크기를 맞춰주는 함수
    pre_transfn = A.Resize(args.image_size, args.image_size)

    if args.train_key in ["mask", "age"]:
        pre_transfn = A.CenterCrop(args.image_size, args.image_size, p=1)

    train_transform = A.Compose(
        [pre_transfn, *new_trans_fns, A.Normalize(mean, std)]  # keep uint8
    )

    test_transform = A.Compose([pre_transfn, A.Normalize(mean, std)])
    train_transform = partial(train_transform, mask=mask)

    return train_transform, test_transform


class MaskDataSet(Dataset):
    def __init__(self, args, is_train=True, transform=None):
        self.args = args
        csv_file = os.path.join(args.data_dir, "train.csv")
        self.datas = pd.read_csv(csv_file)

        self.images, self.labels = self._load_image_files_path(args, is_train)

        if args.train_key == "age-coral":
            self.label_idx = 1
        else:
            self.label_idx = ["gender", "age", "mask"].index(args.train_key)

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

        #  if self.transform:
        #      img = self.transform(img)

        lbl = self._get_label(idx)

        return img, lbl

    def __len__(self):
        return len(self.images)

    def _get_label(self, idx):
        return self.labels[idx][self.label_idx]

    def _load_dataframe(self, args, is_train):

        if args.valid_size == 0:
            return self.datas

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

        return dataset

    def _mapping_label(self, age_lbl, gender_lbl):
        # ["age < 30", "30 <= age < 60", "60 <= age"]
        gender_class = ["male", "female"].index(gender_lbl)
        age_class = 0

        if age_lbl >= 58:  # TODO: TODO: TODO
            age_class = 2
        elif age_lbl >= 30:
            age_class = 1

        if self.args.train_key == "age-coral":
            age_class = age_lbl - 18

        return age_class, gender_class

    def _load_image_files_path(self, args, is_train):

        dataset = self._load_dataframe(args, is_train)
        gender_classes = ["male", "female"]

        images, labels = [], []

        for dir_name in dataset["path"]:
            dir_path = os.path.join(args.data_dir, "images", dir_name)

            image_id, gender_lbl, _, age_lbl = dir_name.split("_")
            age_class, gender_class = self._mapping_label(int(age_lbl), gender_lbl)

            for jpg_filepath in glob(dir_path + "/*"):
                jpg_basename = os.path.basename(jpg_filepath)

                mask_class = 0

                if "incorrect" in jpg_basename:
                    mask_class = 1
                elif "normal" in jpg_basename:
                    mask_class = 2

                if args.use_only_mask and mask_class == 2:
                    continue

                images.append(jpg_filepath)
                labels.append((gender_class, age_class, mask_class))

        return images, labels


def get_dataloader(args):
    train_transform, test_transform = get_album_transforms(args)
    #  train_transform, test_transform = get_transforms(args)

    train_dataset = MaskDataSet(args, is_train=True, transform=train_transform)
    valid_dataset = MaskDataSet(args, is_train=False, transform=test_transform)

    def callback_get_label(dataset, idx):
        """ dataset: MaskDataSet  """
        return dataset._get_label(idx)

    train_dataloader = DataLoader(
        train_dataset,
        #  shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
        sampler=ImbalancedDatasetSampler(
            train_dataset, callback_get_label=callback_get_label
        ),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
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
