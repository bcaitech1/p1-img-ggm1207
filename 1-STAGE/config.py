import os
import argparse

from functools import partial


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="image classificaions")

    pa = partial(parser.add_argument)

    pa("--aug_keys", type=str, default="CJ,FancyPCA")
    pa("--seed", type=int, default=42, help="random seed")
    pa("--test", type=str2bool, default=True, help="small dataset")
    pa("--inf_filename", type=str, default="valid", help="inference filename")
    pa("--data_dir", type=str, default="/opt/ml/input/data/train")
    pa("--valid_size", type=float, default=0.5, help="valid rate")
    pa("--age_model", type=str, default="age", help="small dataset")
    pa("--model_save", type=str2bool, default=True, help="model save??")
    pa("--optimizer", type=str, default="adam", help="small dataset")
    pa("--batch_size", default=64, type=int, help="input batch size")
    pa("--mask_model", type=str, default="mask", help="small dataset")
    pa("--image_size", default=224, type=int, help="the height/width")
    pa("--gender_model", type=str, default="gender", help="small dataset")
    pa("--epochs", type=int, default=25, help="number of epochs to train for")
    pa("--workers", type=int, default=2, help="number of data loading workers")
    pa("--lr", type=float, default=0.001, help="learning rate, default=0.0002")
    pa("--use_only_mask", type=str2bool, default=False, help="mask쓴 데이터만 구성")
    pa("--loss_metric", type=str, default="f1_loss", help="로스 설정")
    pa("--eval", type=str2bool, default=True, help="validation 하는지")
    pa(
        "--train_key",
        type=str,
        default="mask",
        help="split key in ['gender', 'age', 'mask']",
    )
    pa(
        "--model_path",
        type=str,
        default="/opt/ml/weights/",
        help="path of model's weights",
    )

    # args = [] 주면 sweep 작동 안함...
    args, unknown = parser.parse_known_args()

    #  args.lr = args.lr * args.batch_size / 256

    age_model_path = "kage" if args.use_only_mask else args.age_model

    args.age_model = os.path.join(args.model_path, f"{age_model_path}.pt")
    args.gender_model = os.path.join(args.model_path, f"{args.gender_model}.pt")
    args.mask_model = os.path.join(args.model_path, f"{args.mask_model}.pt")

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    return args
