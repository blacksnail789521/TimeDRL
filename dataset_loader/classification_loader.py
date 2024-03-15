import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split


def arg_setup_classification(args):
    # Set args based on data_name for forecasting
    if args.data_name == "HAR":
        args.C = 9
        args.K = 6
        args.seq_len = 128
    elif args.data_name == "WISDM":
        args.C = 3
        args.K = 6
        args.seq_len = 256
    elif args.data_name == "Epilepsy":
        args.C = 1
        args.K = 2
        args.seq_len = 178
    elif args.data_name == "PenDigits":
        args.C = 2
        args.K = 10
        args.seq_len = 8
    elif args.data_name == "FingerMovements":
        args.C = 28
        args.K = 2
        args.seq_len = 50
    else:
        raise NotImplementedError(
            f"Data name '{args.data_name}' is not implemented for classification task."
        )

    return args


class Dataset_Classification(Dataset):
    def __init__(self, dataset, means, stds, percent):
        super().__init__()

        self.x = dataset["samples"]  # (N, T, C)
        self.y = dataset["labels"]  # (N,)

        # Normalize
        self.x = (self.x - means) / stds

        # Select a subset of the dataset
        if percent < 100:
            num_samples = int(self.x.shape[0] * percent / 100)
            self.x = self.x[:num_samples]
            self.y = self.y[:num_samples]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def load_all_datasets(args, mode, dataset_folder):
    # Get percent based on mode
    if mode == "pretrain":
        percent = getattr(args, "pretrain_data_percent", 100)
    elif mode == "linear_eval":
        percent = getattr(args, "linear_eval_data_percent", 100)
    else:
        raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    # Check if validation dataset exists
    valid_dataset_path = dataset_folder / "val.pt"
    valid_dataset_exists = valid_dataset_path.exists()

    # Load datasets
    train_dataset = torch.load(dataset_folder / "train.pt")
    if valid_dataset_exists:
        valid_dataset = torch.load(valid_dataset_path)
    else:
        # # Split training dataset into training and validation
        # X_train = train_dataset["samples"]
        # y_train = train_dataset["labels"]
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train, y_train, test_size=0.2, random_state=42
        # )
        # train_dataset = {"samples": X_train, "labels": y_train}
        # valid_dataset = {"samples": X_val, "labels": y_val}

        # Use test dataset as validation dataset
        valid_dataset = torch.load(dataset_folder / "test.pt")
    test_dataset = torch.load(dataset_folder / "test.pt")

    # Extract mean and std from training dataset
    X_train = train_dataset["samples"]  # (B, T, C)
    means = X_train.mean(dim=[0, 1])  # (C,)
    stds = X_train.std(dim=[0, 1])  # (C,)

    # Preprocess datasets (normalization)
    train_dataset = Dataset_Classification(train_dataset, means, stds, percent)
    valid_dataset = Dataset_Classification(valid_dataset, means, stds, percent)
    test_dataset = Dataset_Classification(test_dataset, means, stds, percent)

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,  # Don't drop last for classification
        # num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        # num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        # num_workers=args.num_workers,
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )
