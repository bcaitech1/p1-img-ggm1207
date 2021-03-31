import numpy as np

import torch.nn.functional as F
import matplotlib.pyplot as plt

from prepare import get_classes


def plot_image(ax, image, output, pred_label, true_label, classes):
    ax.grid(False)

    color = "red"
    if pred_label == true_label:
        color = "blue"

    ax.imshow(image)
    ax.set_xlabel(
        "{} {:2.0f}% ({})".format(
            classes[pred_label], 100 * output[pred_label], classes[true_label]
        ),
        color=color,
    )


def plot_value_array(ax, output, pred_label, true_label, classes):
    ax.grid(False)
    ax.set_ylim([0, 1])

    thisplot = ax.bar(range(len(classes)), output, color="#777777")

    thisplot[pred_label].set_color("red")
    thisplot[true_label].set_color("blue")


def plots_result(args, images, outputs, labels):
    """ all inputs are tensor """

    MEAN = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    STD = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    outputs = F.softmax(outputs, dim=1)

    # to numpy
    images = images.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # image preprocessing
    images = np.clip((images * STD) + MEAN, 0, 1)  # 0 ~ 1
    images = images.transpose(0, 2, 3, 1)  # (batch, width, height, channel)

    classes = get_classes(args.train_key)

    num_rows = num_cols = int(len(images) ** 0.5)
    num_images = num_rows * num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols * 2, figsize=(36, 36))
    plt.setp(axes, xticks=[], yticks=[])

    for idx in range(num_images):
        image, output, label = images[idx], outputs[idx], labels[idx]

        num_row = idx // num_rows
        num_col = idx % num_cols

        pred_label = np.argmax(output)
        true_label = label

        plot_image(
            axes[num_row][num_col * 2], image, output, pred_label, true_label, classes
        )

        plot_value_array(
            axes[num_row][num_col * 2 + 1], output, pred_label, true_label, classes
        )

    return fig
