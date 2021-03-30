import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="DCGAN")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument(
        "--workers", type=int, default=6, help="number of data loading workers"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="input batch size")
    parser.add_argument(
        "--image_size",  # 이미지 사이즈를 지정해놓으니 모델 구현하기가 편한다.
        default=224,
        type=int,
        help="the height/width of the input image to network",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="number of epochs to train for"
    )
    parser.add_argument("--valid_size", type=float, default=0.2, help="valid rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--train_key",
        type=str,
        default="mask",
        help="split key in ['gender', 'age', 'mask']",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/opt/ml/weights/",
        help="path of model's weights",
    )
    parser.add_argument("--manual_seed", type=int, default=42, help="manual seed")
    parser.add_argument("--test", type=bool, default=True, help="small dataset")
    parser.add_argument("--optimizer", type=str, default="adam", help="small dataset")
    parser.add_argument("--age_model", type=str, default="age.pt", help="small dataset")
    parser.add_argument("--gender_model", type=str, default="gender.pt", help="small dataset")
    parser.add_argument("--mask_model", type=str, default="mask.pt", help="small dataset")

    args, unknown = parser.parse_known_args(args=[])

    args.age_model = os.path.join(args.model_path, args.age_model)
    args.gender_model = os.path.join(args.model_path, args.gender_model)
    args.mask_model = os.path.join(args.model_path, args.mask_model)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    return args
