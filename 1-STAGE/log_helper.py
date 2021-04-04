import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from prepare import get_classes
from metrics import apply_grad_cam_pp_to_images, tensor_images_to_numpy_images


def log_f1_and_acc_scores(args, labels, outputs):
    # class 별 f1_score를 계산해야함.

    classes = get_classes(args)
    summary_table = pd.DataFrame([])

    for class_idx in range(len(classes)):
        fancy_idx = np.where(labels == class_idx)

        binary_labels = labels[fancy_idx] == class_idx
        binary_outputs = outputs[fancy_idx] == class_idx

        f1 = f1_score(binary_labels, binary_outputs, average="binary")
        pr = precision_score(binary_labels, binary_outputs, average="binary")
        re = recall_score(binary_labels, binary_outputs, average="binary")
        acc = accuracy_score(binary_labels, binary_outputs)

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
    images_per_row = min(len(instances), images_per_row)
    images = [instance for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty, 3)))

    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    ax.imshow(image)


def log_confusion_matrix_by_images(args, model, images, labels, preds):
    """
    images: tensor for grad_cam
    labels: numpy
    preds: numpy
    """
    classes = get_classes(args)
    cnum = len(classes)

    fig, axes = plt.subplots(nrows=cnum, ncols=cnum, figsize=(36, 36))

    for idx, (l_idx, p_idx) in enumerate(itertools.product(range(cnum), range(cnum))):
        conf_images = images[(labels == l_idx) & (preds == p_idx)]
        conf_images = apply_grad_cam_pp_to_images(args, model, conf_images)
        conf_images = tensor_images_to_numpy_images(conf_images, renormalize=False)
        try:
            _log_confusion_matrix_by_images(
                args, axes[idx // cnum][idx % cnum], conf_images[:25], images_per_row=5
            )
        except Exception as e:
            print(e)

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


def plots_result(args, images, labels, outputs, title="plots_result"):
    """ all inputs are numpy """

    outputs = softmax(outputs, axis=1)
    classes = get_classes(args)

    num_rows = num_cols = int(len(images) ** 0.5)
    num_images = num_rows * num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols * 2, figsize=(36, 18))
    fig.suptitle(title, fontsize=54)

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
