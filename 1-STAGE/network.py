import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet34
from coral_pytorch.layers import CoralLayer


class ResNetClassification(nn.Module):
    def __init__(self, args, num_class):
        super(ResNetClassification, self).__init__()

        self.backbone = resnet34(pretrained=True)

        if args.loss_metric == "coral_loss":
            self.backbone.fc = CoralLayer(self.backbone.fc.in_features, num_class)
        else:
            self.backbone.fc = nn.Linear(
                self.backbone.fc.in_features, num_class, bias=True
            )

    def forward(self, X):
        return self.backbone(X)


def get_resnet34(args, num_class):
    resnet = resnet34(pretrained=True)

    if args.loss_metric == "coral_loss":
        resnet.fc = CoralLayer(resnet.fc.in_features, num_class)
    else:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_class, bias=True)

    return resnet
