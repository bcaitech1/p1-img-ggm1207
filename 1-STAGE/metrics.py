from albumentations.augmentations.functional import get_num_channels
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score

from gradcam import GradCAMpp
from prepare import get_classes, get_num_classes
from gradcam.utils import visualize_cam
from coral_pytorch.losses import coral_loss

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


def tensor_images_to_numpy_images(images, renormalize=False):
    images = images.detach().cpu().numpy()
    if renormalize:
        images = np.clip((images * STD) + MEAN, 0, 1)
    images = images.transpose(0, 2, 3, 1)
    return images


def apply_grad_cam_pp_to_images(args, model, images):
    gradcam_pp = GradCAMpp.from_config(
        model_type="resnet", arch=model, layer_name="layer4"
    )

    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(args.device).reshape(-1, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(args.device).reshape(-1, 1, 1)

    new_images = torch.empty_like(images).to(args.device)
    images = torch.clamp((images * STD) + MEAN, 0, 1)

    for idx, image in enumerate(images):
        mask_pp, _ = gradcam_pp(image.unsqueeze(0))
        heatmap_pp, result_pp = visualize_cam(mask_pp, image)
        new_images[idx] = result_pp

    return new_images


def tensor_to_numpy(tensors):
    return tensors.detach().cpu().numpy()


def get_optimizers(args, model):
    optim_fn = optim.Adam
    if args.optimizer == "adamw":
        optim_fn = optim.AdamW
    if args.optimizer == "sgd":
        optim_fn = optim.SGD
    return optim_fn(model.parameters(), lr=args.lr, weight_decay=0.0001)


def get_lossfn(args):
    num_classes = get_num_classes(args)
    print(num_classes)

    loss_fns = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "f1_loss": F1Loss(classes=num_classes),
        "focal_loss": FocalLoss(gamma=3),
        "smoothing": LabelSmoothingLoss(classes=num_classes, smoothing=0.1),
        "coral_loss": coral_loss,
    }

    loss_fn = loss_fns[args.loss_metric]

    return loss_fn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


#  class FocalLoss(nn.Module):
#      def __init__(self, weight=None, gamma=2.0, reduction="mean"):
#          nn.Module.__init__(self)
#          self.weight = weight
#          self.gamma = gamma
#          self.reduction = reduction
#
#      def forward(self, input_tensor, target_tensor):
#          log_prob = F.log_softmax(input_tensor, dim=-1)
#          prob = torch.exp(log_prob)
#          return F.nll_loss(
#              ((1 - prob) ** self.gamma) * log_prob,
#              target_tensor,
#              weight=self.weight,
#              reduction=self.reduction,
#          )


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
