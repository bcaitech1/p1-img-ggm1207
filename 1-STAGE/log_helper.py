import itertools
import numpy as np
import pandas as pd

import torch.nn.functional as F
import matplotlib.pyplot as plt

from prepare import get_classes
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def log_f1_and_acc_scores(args, labels, outputs):
    # class 별 f1_score를 계산해야함.

    classes = get_classes(args)
    summary_table = pd.DataFrame([])

    for class_idx in range(len(classes)):
        fancy_idx = np.where(labels == class_idx)

        f1 = f1_score(labels[fancy_idx], outputs[fancy_idx], average="macro")
        pr = precision_score(labels[fancy_idx], outputs[fancy_idx], average="macro")
        re = recall_score(labels[fancy_idx], outputs[fancy_idx], average="macro")
        acc = accuracy_score(labels[fancy_idx], outputs[fancy_idx])

        summary_table.loc[args.train_key, f"{class_idx} f1"] = f1
        summary_table.loc[args.train_key, f"{class_idx} pr"] = pr
        summary_table.loc[args.train_key, f"{class_idx} re"] = re
        summary_table.loc[args.train_key, f"{class_idx} acc"] = acc

    return summary_table


def log_confusion_matrix(args, labels, preds):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    fig.suptitle("Confusion Matrix", fontsize=16)
    cmap = plt.cm.Oranges

    cm = confusion_matrix(labels, preds)
    classes = get_classes(args)

    axes[0].imshow(cm, interpolation="nearest", cmap=cmap)

    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    np.fill_diagonal(cm, 0)
    axes[1].imshow(cm, interpolation="nearest", cmap=cmap)

    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(classes)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Predicted label")

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1].text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    return fig


def _log_confusion_matrix_by_images(args, ax, instances, images_per_row=10):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    size = args.image_size
    image_per_row = min(len(instances), images_per_row)
    images = [instance for instance in instances]
    n_rows = (len(instances) - 1) // image_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty, 3)))

    for row in range(n_rows):
        rimages = images[row * image_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    ax.imshow(image)


def log_confusion_matrix_by_images(args, images, labels, preds):
    classes = get_classes(args)
    cnum = len(classes)

    fig, axes = plt.subplots(nrows=cnum, ncols=cnum, figsize=(36, 36))

    for idx, (l_idx, p_idx) in enumerate(itertools.product(range(cnum), range(cnum))):
        conf_images = images[(labels == l_idx) & (preds == p_idx)]
        try:
            _log_plots_image(args, axes[idx], conf_images[:25], images_per_row=5)
        except:
            pass

    return fig


# Below Use train.py
def _log_plots_image(ax, image, output, pred_label, true_label, classes):
    ax.grid(False)
    color = "blue" if pred_label == true_label else "red"

    ax.imshow(image)
    ax.set_xlabel(
        "{} {:2.0f}% ({})".format(
            classes[pred_label], 100 * output[pred_label], classes[true_label]
        ),
        color=color,
    )


def _log_plots_distribution(ax, output, pred_label, true_label, classes):
    ax.grid(False)
    ax.set_ylim([0, 1])

    thisplot = ax.bar(range(len(classes)), output, color="#777777")

    thisplot[pred_label].set_color("red")
    thisplot[true_label].set_color("blue")


def plots_result(args, images, outputs, labels):
    """ all inputs are numpy """

    #  MEAN = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    #  STD = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    outputs = np.softmax(outputs, dim=1)
    classes = get_classes(args)

    num_rows = num_cols = int(len(images) ** 0.5)
    num_images = num_rows * num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols * 2, figsize=(36, 18))
    plt.setp(axes, xticks=[], yticks=[])

    for idx in range(num_images):
        image, output, label = images[idx], outputs[idx], labels[idx]

        num_row = idx // num_rows
        num_col = idx % num_cols

        pred_label = np.argmax(output)
        true_label = label

        _log_plots_image(
            axes[num_row][num_col * 2], image, output, pred_label, true_label, classes
        )

        _log_plots_distribution(
            axes[num_row][num_col * 2 + 1], output, pred_label, true_label, classes
        )

    return fig
