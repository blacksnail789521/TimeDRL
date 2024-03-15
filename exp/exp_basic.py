import os
import torch
from models import TimeDRL


class Exp_Basic(object):
    def __init__(self, args):
        args = self.set_args(args)
        self.args = args
        self.model_dict = {
            "TimeDRL": TimeDRL,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def set_args(self, args):
        # Set d_model
        args.d_model = args.base_d_model * args.n_heads
        print("### Set d_model to {}".format(args.d_model))

        # Set T_p (forecasting)
        args.T_p = int((args.seq_len - args.patch_len) / args.stride + 2)

        # Set i_dim (contrastive)
        if args.get_i == "cls":
            args.i_dim = args.d_model
        elif args.get_i == "last":
            args.i_dim = args.d_model
        elif args.get_i == "gap":
            args.i_dim = args.d_model
        elif args.get_i == "all":
            args.i_dim = args.T_p * args.d_model

        # Set patience
        if args.task_name == "forecasting":
            args.patience = 5
        elif args.task_name == "classification":
            args.patience = 10

        return args

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # Assume we only use 1 gpu at most
            print("Use GPU")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
