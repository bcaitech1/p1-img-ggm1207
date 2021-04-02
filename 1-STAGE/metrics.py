import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score

MEAN = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)


def change_2d_to_1d(tens):
    if len(tens.shape) == 2:
        tens = tens.reshape(-1)
    return tens


#  def change_age_to_cat(age_logit):
#      """ age_logit: 1d tensor """
#      age_logit = torch.where(age_logit < 30, torch.zeros_like(age_logit), age_logit)
#      age_logit = torch.where(age_logit >= 60, torch.ones_like(age_logit) * 2, age_logit)
#      age_logit = torch.where(age_logit >= 30, torch.ones_like(age_logit), age_logit)
#      return age_logit.type(torch.long)


def calulate_18class(mi, gi, ai):
    return 6 * mi + 3 * gi + ai


def tensor_images_to_numpy_images(images):
    images = images.detach().cpu().numpy()
    images = np.clip((images * STD) + MEAN, 0, 1)
    return images


def tensor_to_numpy(tensors):
    return tensors.detach().cpu().numpy()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()

        return loss.sum()
