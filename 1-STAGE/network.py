import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet34


class ResNetClassification(nn.Module):
    def __init__(self, num_class):
        super(ResNetClassification, self).__init__()

        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_class, bias=True)

    def forward(self, X):
        return self.backbone(X)


def get_resnet34(num_class):
    resnet = resnet34(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_class, bias=True)
    return resnet
