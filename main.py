import argparse
import os
from pathlib import Path
import sys
import torch
import numpy as np
from copy import deepcopy
import shutil
import time

from exp.exp_forecasting import Exp_Forecasting
from exp.exp_classification import Exp_Classification
from utils.tools import (
    set_seed,
    print_formatted_dict,
)
from dataset_loader.dataset_loader import update_args_from_dataset


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimeDRL")

    # * basic config
    parser.add_argument(
        "--task_name",
        type=str,
        default="forecasting",
        choices=["forecasting", "classification"],
        help="time-series tasks",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TimeDRL",
        choices=["TimeDRL"],
        help="model name",
    )
    parser.add_argument(
        "--train_together",
        action="store_true",
        help="train pretrain and linear_eval together (udr this for semi-supervised learning) "
        "the purpose of this is to get the good encoder for downstream task",
        default=False,
    )
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
    )
    parser.add_argument(
        "--delete_checkpoints",
        action="store_true",
        help="delete checkpoints after training",
        # default=False,
        default=True,
    )

    # * data loader
    parser.add_argument(
        "--data_name",
        type=str,
        default="ETTh1",
        choices=[
            # ? forecasting
            "ETTh1",
            "ETTh2",
            "ETTm1",
            "ETTm2",
            "Exchange",
            "Weather",
            # ? classification
            "HAR",
            "WISDM",
            "Epilepsy",
            "PenDigits",
            "FingerMovements",
        ],
        help="data name",
    )
    parser.add_argument(
        "--pred_len_list",
        type=int,
        nargs="+",
        default=[24, 48, 168, 336, 720],
        help="prediction sequence length list (only for forecasting)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        choices=["M", "S"],
        help="multivariate or univariate (only for forecasting)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--data_aug",
        type=str,
        default="none",
        choices=[
            "none",
            "jitter",
            "scaling",
            "rotation",
            "permutation",
            "masking",
            "cropping",
        ],
        help="data augmentation",
    )
    parser.add_argument(
        "--pretrain_data_percent",
        type=int,
        default=100,
        help="percentage of pretrain data (only for semi-supervised learning)",
    )
    parser.add_argument(
        "--linear_eval_data_percent",
        type=int,
        default=100,
        help="percentage of linear_eval data (only for semi-supervised learning)",
    )

    # * forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # * embedding
    parser.add_argument(
        "--pos_embed_type",
        type=str,
        default="fixed",
        choices=["fixed", "learnable", "none"],
        help="position embedding type",
    )
    parser.add_argument(
        "--token_embed_type",
        type=str,
        default="linear",
        choices=["linear", "conv"],
        help="token embedding type",
    )
    parser.add_argument(
        "--token_embed_kernel_size",
        type=int,
        default=3,
        help="kernel size of token embedding convolution",
    )

    # * model architecture
    parser.add_argument(
        "--encoder_arch",
        type=str,
        default="transformer_encoder",
        choices=[
            "transformer_encoder",
            "transformer_decoder",
            "resnet",
            "tcn",
            "lstm",
            "bilstm",
        ],
        help="encoder architecture",
    )
    parser.add_argument(
        "--get_i",
        type=str,
        default="cls",
        choices=["cls", "last", "gap", "all"],
        help="the way to get the instance-level representation",
    )
    parser.add_argument(
        "--base_d_model",
        type=int,
        default=64,
        help="the base number of embedding dimension (d_model = base_d_model * n_heads)",
    )
    parser.add_argument("--n_layers", type=int, default=12, help="number of layers")
    parser.add_argument("--n_heads", type=int, default=12, help="number of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="activation function",
        choices=["relu", "gelu"],
    )
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--enable_channel_independence",
        type=bool,
        default=True,
        help="enable channel independence",
    )

    # * training_stage_params (pretrain)
    parser.add_argument(
        "--pretrain_optim",
        type=str,
        default="AdamW",
        help="optimizer (pretrain)",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--pretrain_learning_rate",
        type=float,
        default=0.001,
        help="learning rate (pretrain)",
    )
    parser.add_argument(
        "--pretrain_lradj",
        type=str,
        default="type1",
        help="adjust learning rate (pretrain)",
    )
    parser.add_argument(
        "--pretrain_weight_decay",
        type=float,
        default=0.001,
        help="l2 weight decay (pretrain)",
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10, help="epochs for training (pretrain)"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
        help="weight for contrastive loss (aka lambda)",
    )
    parser.add_argument(
        "--disable_predictive_loss",
        action="store_true",
        help="disable predictive loss",
        default=False,
    )
    parser.add_argument(
        "--disable_contrastive_loss",
        action="store_true",
        help="disable contrastive loss",
        default=False,
    )
    parser.add_argument(
        "--disable_stop_gradient",
        action="store_true",
        help="disable stop gradient",
        default=False,
    )
    parser.add_argument(
        "--disable_freeze_encoder",
        action="store_true",
        help="disable freeze encoder (fine-tune the encoder, only for semi-supervised learning)",
        default=False,
    )

    # * training_stage_params (linear_eval)
    parser.add_argument(
        "--linear_eval_optim",
        type=str,
        default="AdamW",
        help="optimizer (linear_eval)",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--linear_eval_learning_rate",
        type=float,
        default=0.001,
        help="learning rate (linear_eval)",
    )
    parser.add_argument(
        "--linear_eval_lradj",
        type=str,
        default="type1",
        help="adjust learning rate (linear_eval)",
    )
    parser.add_argument(
        "--linear_eval_weight_decay",
        type=float,
        default=0.001,
        help="l2 weight decay (linear_eval)",
    )
    parser.add_argument(
        "--linear_eval_epochs",
        type=int,
        default=10,
        help="epochs for training (linear_eval)",
    )

    # * training_stage_params (shared)
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument(
        "--delta", type=float, default=0.0001, help="early stopping delta"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        # default=False,
        default=True,  # faster
    )

    # * GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")

    args, _ = parser.parse_known_args()

    # * Not used
    args.target = "OT"
    args.label_len = 0
    args.freq = "h"
    args.return_single_feature = False

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in tunable_params.items():
        if key == "seq_len" and args.task_name == "classification":
            # Skip seq_len for classification
            continue
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args(
    args: argparse.Namespace,
    fixed_params: dict,
    tunable_params: dict,
) -> argparse.Namespace:
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
    args = update_args_from_dataset(args)

    args.setting = f"{args.task_name}_{args.data_name}"
    print(f"Args in experiment: {args}")

    return args


def run_exp(args: argparse.Namespace) -> dict:
    # * Get exp
    if args.task_name == "forecasting":
        exp = Exp_Forecasting(args)
    elif args.task_name == "classification":
        exp = Exp_Classification(args)
    else:
        raise NotImplementedError

    # * Run the experiment
    if args.train_together:
        metrics = exp.train_together(use_tqdm=True)
    else:
        metrics = exp.train(use_tqdm=True)

    # * Delete checkpoints
    if args.delete_checkpoints:
        shutil.rmtree(args.checkpoints, ignore_errors=True)

    return metrics


def trainable_forecasting(
    args: argparse.Namespace,
) -> dict:
    # Run experiments with different pred_len
    metrics_dict = {}
    for pred_len in args.pred_len_list:
        # Update pred_len
        args.pred_len = pred_len

        metrics_dict[pred_len] = run_exp(args)

    # Return metrics
    return_metrics = {}
    return_metrics["avg_mse"] = np.mean(
        [v["best_test_mse"] for v in metrics_dict.values()]
    )
    return_metrics["avg_mae"] = np.mean(
        [v["best_test_mae"] for v in metrics_dict.values()]
    )
    for pred_len in args.pred_len_list:
        return_metrics[f"{pred_len}_mse"] = metrics_dict[pred_len]["best_test_mse"]
        return_metrics[f"{pred_len}_mae"] = metrics_dict[pred_len]["best_test_mae"]

    return (
        return_metrics  # We only care about the best test loss for the downstream task
    )


def trainable_classification(
    args: argparse.Namespace,
) -> dict:
    # Run the experiment
    metrics = run_exp(args)

    return metrics  # We only care about the best test loss for the downstream task


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
) -> dict:
    # Update args
    args = update_args(args, fixed_params, tunable_params)

    if fixed_params["task_name"] == "forecasting":
        return_metrics = trainable_forecasting(args)
    elif fixed_params["task_name"] == "classification":
        return_metrics = trainable_classification(args)
    else:
        raise NotImplementedError

    return return_metrics


if __name__ == "__main__":
    """------------------------------------"""
    task_name = "forecasting"
    # task_name = "classification"

    if task_name == "forecasting":
        # * data_name
        data_name = "ETTh1"  # 7
        # data_name = "ETTh2"  # 7
        # data_name = "ETTm1"  # 7
        # data_name = "ETTm2"  # 7
        # data_name = "Exchange"  # 8
        # data_name = "Weather"  # 21

        # * pred_len_list
        pred_len_list = [24, 48, 168, 336, 720]  # ETTh1, ETTh2, Exchange, Weather
        # pred_len_list = [24, 48, 96, 228, 672]  # ETTm1, ETTm2
        # pred_len_list = [168]  # ablation study

        # * features
        features = "M"  # Multivariate
        # features = "S"  # Univariate
    elif task_name == "classification":
        # * data_name
        # data_name = "HAR"  # (128, 9)
        data_name = "WISDM"  # (256, 3)
        # data_name = "Epilepsy"  # (178, 1)
        # data_name = "PenDigits"  # (8, 2)
        # data_name = "FingerMovements"  # (50, 28)
    else:
        raise NotImplementedError

    num_workers = 4  # It doesn't really matter since we don't use it in dataloader
    # num_workers = 6
    # num_workers = 8

    # batch_size = 8
    batch_size = 32
    # batch_size = 128  # 8G
    # batch_size = 512 # 24G
    """------------------------------------"""
    train_together = False  # default
    # train_together = True  # Use this for semi-supervised learning

    get_i = "cls"  # default
    # get_i = "last"
    # get_i = "gap"
    # get_i = "all"

    encoder_arch = "transformer_encoder"  # default
    # encoder_arch = "transformer_decoder"
    # encoder_arch = "resnet"
    # encoder_arch = "tcn"
    # encoder_arch = "lstm"
    # encoder_arch = "bilstm"

    data_aug = "none"  # default
    # data_aug = "jitter"
    # data_aug = "scaling"
    # data_aug = "rotation"
    # data_aug = "permutation"
    # data_aug = "masking"
    # data_aug = "cropping"

    disable_stop_gradient = False  # default
    # disable_stop_gradient = True

    disable_predictive_loss = False  # default
    # disable_predictive_loss = True

    disable_contrastive_loss = False  # default
    # disable_contrastive_loss = True

    pretrain_data_percent = 100  # default
    # pretrain_data_percent = 75
    # pretrain_data_percent = 50
    # pretrain_data_percent = 10
    # pretrain_data_percent = 5
    # pretrain_data_percent = 1

    linear_eval_data_percent = 100  # default
    # linear_eval_data_percent = 75
    # linear_eval_data_percent = 50
    # linear_eval_data_percent = 10
    # linear_eval_data_percent = 5
    # linear_eval_data_percent = 1

    disable_freeze_encoder = False  # default
    # disable_freeze_encoder = True
    """------------------------------------"""
    set_seed(seed=2023)

    # Setup args
    args = get_args_from_parser()
    args.root_folder = Path.cwd()  # Set this outside of the trainable function

    # Setup fixed params
    fixed_params = {
        "task_name": task_name,
        "data_name": data_name,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "train_together": train_together,
        "get_i": get_i,
        "encoder_arch": encoder_arch,
        "data_aug": data_aug,
        "disable_stop_gradient": disable_stop_gradient,
        "disable_predictive_loss": disable_predictive_loss,
        "disable_contrastive_loss": disable_contrastive_loss,
        "pretrain_data_percent": pretrain_data_percent,
        "linear_eval_data_percent": linear_eval_data_percent,
        "disable_freeze_encoder": disable_freeze_encoder,
    }
    if task_name == "forecasting":
        fixed_params["pred_len_list"] = pred_len_list  # type: ignore
        fixed_params["features"] = features  # type: ignore

    # Setup tunable params
    # TODO: copy `config` from `exp_settings_and_results` (be careful with the boolean values)
    tunable_params = {
        "pretrain_optim": "AdamW",
        "pretrain_learning_rate": 0.000021350270238434573,
        "pretrain_lradj": "type1",
        "pretrain_weight_decay": 0.005551156847369134,
        "pretrain_epochs": 10,
        "contrastive_weight": 0.5,
        "linear_eval_optim": "AdamW",
        "linear_eval_learning_rate": 0.00021350270238434574,
        "linear_eval_lradj": "warmup",
        "linear_eval_weight_decay": 0.000016413821615923124,
        "linear_eval_epochs": 30,
        "pos_embed_type": "learnable",
        "token_embed_type": "conv",
        "token_embed_kernel_size": 3,
        "dropout": 0.2,
        "base_d_model": 64,
        "n_layers": 1,
        "n_heads": 2,
        "patch_len": 16,
        "stride": 16,
        "enable_channel_independence": True,
        "seq_len": 512,
    }

    # Run
    return_metrics = trainable(tunable_params, fixed_params, args)
    print_formatted_dict(return_metrics)
