import pandas as pd
from pathlib import Path


def load_raw_data(data_dir):
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])

    return train, test, stores, features