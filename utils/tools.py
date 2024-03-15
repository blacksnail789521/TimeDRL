import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import sys
import os
from functools import wraps
from pathlib import Path
import re

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, lradj, learning_rate):
    # Setup lr_adjust
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "type3":
        lr_adjust = {
            epoch: (
                learning_rate
                if epoch < 3
                else learning_rate * (0.9 ** ((epoch - 3) // 1))
            )
        }
    elif lradj == "constant":
        lr_adjust = {epoch: learning_rate}
    elif lradj == "warmup":
        if epoch < 5:  # increase lr for first 5 epochs
            lr_adjust = {epoch: learning_rate * (epoch + 1) / 5}
        else:  # decrease lr for the rest
            lr_adjust = {epoch: learning_rate * (0.9 ** ((epoch - 5) // 1))}
    else:
        raise NotImplementedError

    # Use lr_adjust to update learning rate on certain epochs
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


# def adjust_contrastive_weight(epoch, cwadj, contrastive_weight):
#     # Setup cw_adjust
#     if cwadj == "type1":
#         cw_adjust = {epoch: contrastive_weight * (0.5 ** ((epoch - 1) // 1))}
#     elif cwadj == "type2":
#         cw_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
#     elif cwadj == "type3":
#         cw_adjust = {
#             epoch: contrastive_weight
#             if epoch < 3
#             else contrastive_weight * (0.9 ** ((epoch - 3) // 1))
#         }
#     elif cwadj == "constant":
#         cw_adjust = {epoch: contrastive_weight}
#     elif cwadj == "warmup":
#         if epoch < 5:  # increase lr for first 5 epochs
#             cw_adjust = {epoch: contrastive_weight * (epoch + 1) / 5}
#         else:  # decrease lr for the rest
#             cw_adjust = {epoch: contrastive_weight * (0.9 ** ((epoch - 5) // 1))}
#     else:
#         raise NotImplementedError

#     # Use cw_adjust to update learning rate on certain epochs
#     if epoch in cw_adjust.keys():
#         cw = cw_adjust[epoch]
#         print("Updating contrastive weight to {}".format(cw))
#         return cw
#     else:
#         return contrastive_weight


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model=None, path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience} "
                f"(best val_loss: {self.val_loss_min:.6f}, current val_loss: {val_loss:.6f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if model is None or path is None:
            return

        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def print_params(model, show_num_params=True, show_arch=False):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params

    if show_num_params:
        print("-----------------------------------------")
        print(f"Total number of parameters: {total_params:,}")
        print(f"Number of trainable parameters: {trainable_params:,}")
        print(f"Percentage of trainable parameters: {trainable_ratio:.2%}")

    if show_arch:
        print("-----------------------------------------")
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "h." in name and "h.0" not in name:
                    continue
                print(name.replace("base_model.model.", "").replace("h.0", "h.x"))
        print("-----------------------------------------")

    return trainable_ratio


def change_dict_to_args(configs):
    args = argparse.Namespace()
    for key, value in configs.items():
        setattr(args, key, value)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_formatted_dict(d):
    for key, value in d.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
