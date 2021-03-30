import torch
from sklearn.metrics import accuracy_score, f1_score


def change_2d_to_1d(tens):
    if len(tens.shape) == 2:
        tens = tens.reshape(-1)
    return tens


def change_age_to_cat(age_logit):
    """ age_logit: 1d tensor """
    age_logit = torch.where(age_logit < 30, torch.zeros_like(age_logit), age_logit)
    age_logit = torch.where(age_logit >= 60, torch.ones_like(age_logit) * 2, age_logit)
    age_logit = torch.where(age_logit >= 30, torch.ones_like(age_logit), age_logit)
    return age_logit.type(torch.long)


def cal_metrics(pred, label, score_fn=f1_score):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    return score_fn(pred, label, average="macro")


def cal_accuracy(pred, label):
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    return accuracy_score(pred, label)
