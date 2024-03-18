import torch
import torch.nn as nn
from torch import optim
import os
from pathlib import Path
import time
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from dataset_loader.dataset_loader import load_forecasting_dataloader
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.tools import print_params, print_formatted_dict
from utils.visual import show_table, show_plot
from models import linear_forecaster as linear_eval
from layers.Embed import Patching

warnings.filterwarnings("ignore")


class Exp_Forecasting(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # 1. set args, model_dict, device into self
        # 2. build model

    def _build_model(self):
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # Feed `args` as `config`
        print_params(model)
        return model

    def _build_linear_eval(self):
        D, C, T_out = self.args.d_model, self.args.C, self.args.pred_len
        self.linear_eval = (
            linear_eval.Model(
                D,
                C,
                self.args.T_p,
                T_out,
                self.args.dropout,
                self.args.enable_channel_independence,
            )
            .float()
            .to(self.device)
        )

    def train_together(self, use_tqdm=False):
        #! The purpose of this function is to find a good encoder for the downstream task
        ### Training ###
        print(
            f">>>>> start training together (forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>"
        )
        """
        For each epoch
        1. Train the encoder along with the linear_eval model
        """

        # Get data
        train_loader, valid_loader, test_loader = load_forecasting_dataloader(self.args)

        # Define the linear_eval model (we've already defined the encoder in the previous step)
        self._build_linear_eval()

        checkpoint_path = Path(self.args.checkpoints, self.args.setting)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        # * Define optimizers and early stopping (for both models)
        shared_optim = getattr(optim, self.args.pretrain_optim)(
            list(self.model.parameters()) + list(self.linear_eval.parameters()),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.pretrain_weight_decay,
        )
        shared_early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # * The whole training process (pretrain + linear_eval)
        shared_history = {
            "train": {"loss": [], "mae": []},
            "valid": {"loss": [], "mae": []},
            "test": {"loss": [], "mae": []},
        }
        shared_epochs = self.args.pretrain_epochs
        for shared_epoch in range(shared_epochs):
            self.model.train()
            self.linear_eval.train()

            iter_data = (
                tqdm(
                    train_loader,
                    desc=f"Epoch {shared_epoch + 1}/{shared_epochs}, Training Loss: {0}",
                )
                if use_tqdm
                else train_loader
            )
            train_losses = []
            for i, (batch_x, batch_y, _, _) in enumerate(iter_data):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ? 1. Zero grad
                shared_optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                    # Linear eval
                    y_pred_1 = self.linear_eval(t_1)
                    y_pred_2 = self.linear_eval(t_2)

                    # Reverse Instance Normalization
                    y_pred_1 = self.model.revin(y_pred_1, "denorm")
                    y_pred_2 = self.model.revin(y_pred_2, "denorm")

                    # ? 3. Calculate loss
                    loss = (
                        nn.MSELoss()(y_pred_1, batch_y)
                        + nn.MSELoss()(y_pred_2, batch_y)
                    ) / 2
                    train_losses.append(loss.item())

                # ? 4. Backward
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(shared_optim)
                scaler.update()

                if use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"Epoch {shared_epoch + 1}/{shared_epochs}, "
                        f"Training Loss: {np.mean(train_losses):.3f}"
                    )

                # if i == 10:
                #     break

            # * At the end of each epoch, we get all the metrics
            print(">>>>> Calculate training metrics >>>>>")
            train_loss, train_mae = self.get_metrics(train_loader, use_tqdm)
            shared_history["train"]["loss"].append(train_loss)
            shared_history["train"]["mae"].append(train_mae)
            print(">>>>> Calculate validation metrics >>>>>")
            valid_loss, valid_mae = self.get_metrics(valid_loader, use_tqdm)
            shared_history["valid"]["loss"].append(valid_loss)
            shared_history["valid"]["mae"].append(valid_mae)
            print(">>>>> Calculate testing metrics >>>>>")
            test_loss, test_mae = self.get_metrics(test_loader, use_tqdm)
            shared_history["test"]["loss"].append(test_loss)
            shared_history["test"]["mae"].append(test_mae)

            # * Show metrics for all the previous epochs
            show_table(shared_history)
            show_plot(shared_history)

            # * Early stopping
            shared_early_stopping(valid_loss)
            if shared_early_stopping.early_stop:
                print("Early stopping")
                break

            # * Adjust learning rate
            adjust_learning_rate(
                shared_optim,
                shared_epoch + 1,
                self.args.pretrain_lradj,
                self.args.pretrain_learning_rate,
            )
            print("------------------------------------------------------------------")

        best_mse_epoch = np.nanargmin(shared_history["test"]["loss"])
        metrics = {
            "best_test_mse": shared_history["test"]["loss"][best_mse_epoch],
            "best_test_mae": shared_history["test"]["mae"][best_mse_epoch],
            "best_shared_epoch": best_mse_epoch + 1,
        }
        print("===============================")
        print_formatted_dict(metrics)
        print("===============================")

        self.spent_time = time.time() - start_time

        return metrics

    def train(self, use_tqdm=False):
        #! The purpose of this function is to train the given good encoder
        ### Training ###
        print(
            f">>>>> start training (forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>"
        )
        """
        For each epoch
        1. (Unfreeze the encoder first) Train the encoder with the pretext tasks
        2. Freeze the encoder
        3. Get the linear_eval model and train it along with the encoder for the downstream task
        """

        # Get data
        train_loader, valid_loader, test_loader = load_forecasting_dataloader(
            self.args, mode="pretrain"
        )
        linear_eval_train_loader, _, _ = load_forecasting_dataloader(
            self.args, mode="linear_eval"
        )

        # Define the linear_eval model (we've already defined the encoder in the previous step)
        self._build_linear_eval()

        # Define the patching layer (for the predictive task)
        patching = Patching(
            self.args.patch_len, self.args.stride, self.args.enable_channel_independence
        )

        checkpoint_path = Path(self.args.checkpoints, self.args.setting)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        # * Define optimizers and early stopping (for both models)
        model_optim = getattr(optim, self.args.pretrain_optim)(
            self.model.parameters(),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.pretrain_weight_decay,
        )
        linear_eval_optim = getattr(optim, self.args.linear_eval_optim)(
            self.linear_eval.parameters(),
            lr=self.args.linear_eval_learning_rate,
            weight_decay=self.args.linear_eval_weight_decay,
        )
        model_early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        linear_eval_early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)  # type: ignore

        # * The whole training process (pretrain + linear_eval)
        pretrain_history = {
            "predictive_loss": [],
            "contrastive_loss": [],
            "pretrain_loss": [],
        }
        linear_eval_history = {"best_test_mse": [], "best_test_mae": []}
        for pretrain_epoch in range(self.args.pretrain_epochs):
            ###! 1. Pretrain ###
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True  # Unfreeze the encoder
            trainable_ratio = print_params(self.model, show_num_params=False)
            assert (
                trainable_ratio == 1.0
            ), f"The encoder is not fully trainable (trainable_ratio: {trainable_ratio})"
            assert (
                self.args.disable_predictive_loss == False
                or self.args.disable_contrastive_loss == False
            ), "Both predictive and contrastive losses are disabled"

            iter_data = (
                tqdm(
                    train_loader,
                    desc=f"[Pretrain] Epoch {pretrain_epoch + 1}/{self.args.pretrain_epochs}, "
                    f"Predictive Loss: {0}, Contrastive Loss: {0}, Pretrain Loss: {0}",
                )
                if use_tqdm
                else train_loader
            )
            predictive_losses, contrastive_losses, pretrain_losses = [], [], []
            for i, (batch_x, _, _, _) in enumerate(iter_data):
                batch_x = batch_x.float().to(self.device)

                # ? 1. Zero grad
                model_optim.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    (
                        _,
                        _,
                        x_pred_1,
                        x_pred_2,
                        i_1,
                        i_2,
                        i_1_pred,
                        i_2_pred,
                    ) = self.model(batch_x)

                    # ? 3. Calculate loss
                    # Predictive task
                    if self.args.disable_predictive_loss:
                        predictive_loss = 0
                    else:
                        predictive_loss = (
                            nn.MSELoss()(x_pred_1, patching(batch_x))
                            + nn.MSELoss()(x_pred_2, patching(batch_x))
                        ) / 2

                    # Contrastive task
                    if self.args.disable_contrastive_loss:
                        contrastive_loss = 0
                    else:
                        if not self.args.disable_stop_gradient:
                            i_1 = i_1.detach()
                            i_2 = i_2.detach()
                        cos_sim = nn.CosineSimilarity(dim=1)
                        contrastive_loss = (
                            -(
                                cos_sim(i_1, i_2_pred).mean()
                                + cos_sim(i_2, i_1_pred).mean()
                            )
                            * 0.5
                        )

                    pretrain_loss = (
                        predictive_loss
                        + self.args.contrastive_weight * contrastive_loss
                    )
                    if self.args.disable_predictive_loss:
                        pretrain_losses.append(0)
                    else:
                        predictive_losses.append(predictive_loss.item())  # type: ignore
                    if self.args.disable_contrastive_loss:
                        contrastive_losses.append(0)
                    else:
                        contrastive_losses.append(contrastive_loss.item())  # type: ignore
                    pretrain_losses.append(pretrain_loss.item())

                # ? 4. Backward
                scaler.scale(pretrain_loss).backward()  # type: ignore
                scaler.step(model_optim)
                scaler.update()

                if use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"[Pretrain] Epoch {pretrain_epoch + 1}/{self.args.pretrain_epochs}, "
                        f"Predictive Loss: {np.mean(predictive_losses):.3f}, "
                        f"Contrastive Loss: {np.mean(contrastive_losses):.3f}, "
                        f"Pretrain Loss: {np.mean(pretrain_losses):.3f}"
                    )

                # if i == 10:
                #     break

            ###! 2. Linear Eval ###
            local_linear_eval_history = {
                # "train": {"loss": [], "mae": []},
                "valid": {"loss": [], "mae": []},
                "test": {"loss": [], "mae": []},
            }

            self.linear_eval.train()
            if not self.args.disable_freeze_encoder:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False  # Freeze the encoder
                trainable_ratio = print_params(self.model, show_num_params=False)
                assert (
                    trainable_ratio == 0.0
                ), f"The encoder is not fully frozen (trainable_ratio: {trainable_ratio})"

            for linear_eval_epoch in range(self.args.linear_eval_epochs):
                iter_data = (
                    tqdm(
                        linear_eval_train_loader,
                        desc=f"({pretrain_epoch + 1}/{self.args.pretrain_epochs}) "
                        f"[Linear Eval] Epoch {linear_eval_epoch + 1}/{self.args.linear_eval_epochs}, Training Loss: {0}",
                    )
                    if use_tqdm
                    else linear_eval_train_loader
                )
                train_losses = []
                for i, (batch_x, batch_y, _, _) in enumerate(iter_data):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    # ? 1. Zero grad
                    linear_eval_optim.zero_grad()

                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                        # ? 2. Call the model
                        # Encoder (including instance normalization)
                        t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                        # Linear eval
                        y_pred_1 = self.linear_eval(t_1)
                        y_pred_2 = self.linear_eval(t_2)

                        # Reverse Instance Normalization
                        y_pred_1 = self.model.revin(y_pred_1, "denorm")
                        y_pred_2 = self.model.revin(y_pred_2, "denorm")

                        # ? 3. Calculate loss
                        loss = (
                            nn.MSELoss()(y_pred_1, batch_y)
                            + nn.MSELoss()(y_pred_2, batch_y)
                        ) / 2
                        train_losses.append(loss.item())

                    # ? 4. Backward
                    scaler.scale(loss).backward()  # type: ignore
                    scaler.step(linear_eval_optim)
                    scaler.update()

                    if use_tqdm:
                        iter_data.set_description(  # type: ignore
                            f"({pretrain_epoch + 1}/{self.args.pretrain_epochs}) "
                            f"[Linear Eval] Epoch {linear_eval_epoch + 1}/{self.args.linear_eval_epochs}, "
                            f"Training Loss: {np.mean(train_losses):.3f}"
                        )

                    # if i == 10:
                    #     break

                # * At the end of each epoch, we get all the metrics
                # print(">>>>> Calculate training metrics >>>>>")
                # train_loss, train_mae = self.get_metrics(train_loader, use_tqdm)
                # local_linear_eval_history["train"]["loss"].append(train_loss)
                # local_linear_eval_history["train"]["mae"].append(train_mae)
                print(">>>>> Calculate validation metrics >>>>>")
                valid_loss, valid_mae = self.get_metrics(
                    valid_loader, use_tqdm
                )  # We need this to do early stopping
                local_linear_eval_history["valid"]["loss"].append(valid_loss)
                local_linear_eval_history["valid"]["mae"].append(valid_mae)
                print(">>>>> Calculate testing metrics >>>>>")
                test_loss, test_mae = self.get_metrics(test_loader, use_tqdm)
                local_linear_eval_history["test"]["loss"].append(test_loss)
                local_linear_eval_history["test"]["mae"].append(test_mae)

                # * Show metrics for all the previous epochs
                show_table(local_linear_eval_history)

                # * Early stopping
                linear_eval_early_stopping(valid_loss)
                if linear_eval_early_stopping.early_stop:
                    print("Early stopping")
                    break

                # * Adjust learning rate
                adjust_learning_rate(
                    linear_eval_optim,
                    linear_eval_epoch + 1,
                    self.args.linear_eval_lradj,
                    self.args.linear_eval_learning_rate,
                )

            # * At the end of each epoch, we get all the metrics (for both pretrain and linear_eval)
            # ? Pretrain
            predictive_loss = np.mean(predictive_losses)
            contrastive_loss = np.mean(contrastive_losses)
            pretrain_loss = np.mean(pretrain_losses)
            pretrain_history["predictive_loss"].append(predictive_loss)
            pretrain_history["contrastive_loss"].append(contrastive_loss)
            pretrain_history["pretrain_loss"].append(pretrain_loss)
            # ? Linear Eval
            if np.all(np.isnan(local_linear_eval_history["test"]["loss"])):
                # All the linear results for this epoch are NaN
                linear_eval_history["best_test_mse"].append(np.nan)
                linear_eval_history["best_test_mae"].append(np.nan)
            else:
                best_mse_epoch = np.nanargmin(local_linear_eval_history["test"]["loss"])
                best_test_mse = local_linear_eval_history["test"]["loss"][
                    best_mse_epoch
                ]
                best_test_mae = local_linear_eval_history["test"]["mae"][best_mse_epoch]
                linear_eval_history["best_test_mse"].append(best_test_mse)
                linear_eval_history["best_test_mae"].append(best_test_mae)

            # * Early stopping
            model_early_stopping(pretrain_loss)
            if model_early_stopping.early_stop:
                print("Early stopping")
                break

            # * Adjust learning rate
            adjust_learning_rate(
                model_optim,
                pretrain_epoch + 1,
                self.args.pretrain_lradj,
                self.args.pretrain_learning_rate,
            )
            print("------------------------------------------------------------------")

        best_pretrain_epoch = np.nanargmin(pretrain_history["pretrain_loss"])
        best_best_test_mse_epoch = np.nanargmin(linear_eval_history["best_test_mse"])
        metrics = {
            # ? Pretrain (train)
            "best_pretrain_loss": pretrain_history["pretrain_loss"][
                best_pretrain_epoch
            ],
            "predictive_loss": pretrain_history["predictive_loss"][best_pretrain_epoch],
            "contrastive_loss": pretrain_history["contrastive_loss"][
                best_pretrain_epoch
            ],
            # ? Linear Eval (test)
            "best_test_mse": linear_eval_history["best_test_mse"][
                best_best_test_mse_epoch
            ],
            "best_test_mae": linear_eval_history["best_test_mae"][
                best_best_test_mse_epoch
            ],
            "best_pretrain_epoch": best_pretrain_epoch + 1,
            "best_best_test_mse_epoch": best_best_test_mse_epoch + 1,
        }
        print("===============================")
        print(f"### pred_len: {self.args.pred_len} ###")
        print_formatted_dict(metrics)
        print("===============================")

        self.spent_time = time.time() - start_time

        return metrics

    def get_metrics(self, data_loader, use_tqdm=False):
        total_mse = 0
        total_mae = 0
        total_samples = 0

        self.model.eval()
        self.linear_eval.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(
                tqdm(data_loader) if use_tqdm else data_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # ? 1. Zero grad
                pass

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    # Encoder (including instance normalization)
                    t_1, t_2, _, _, _, _, _, _ = self.model(batch_x)

                    # Linear eval
                    y_pred_1 = self.linear_eval(t_1)
                    y_pred_2 = self.linear_eval(t_2)

                    # Reverse Instance Normalization
                    y_pred_1 = self.model.revin(y_pred_1, "denorm")
                    y_pred_2 = self.model.revin(y_pred_2, "denorm")

                pred_1 = y_pred_1.detach()
                pred_2 = y_pred_2.detach()
                true = batch_y.detach()

                batch_mse = (
                    torch.mean((pred_1 - true) ** 2).item()
                    + torch.mean((pred_2 - true) ** 2).item()
                ) / 2
                batch_mae = (
                    torch.mean(torch.abs(pred_1 - true)).item()
                    + torch.mean(torch.abs(pred_2 - true)).item()
                ) / 2

                total_mse += batch_mse * len(batch_x)
                total_mae += batch_mae * len(batch_x)
                total_samples += len(batch_x)

                # if i == 10:
                #     break

        mse = total_mse / total_samples
        mae = total_mae / total_samples

        return mse, mae
