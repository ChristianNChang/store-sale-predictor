import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

data_dir = "./store-sales-time-series-forecasting"

def load_data(data_dir):
    # Read the Kaggle train and test CSVs
    full_train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test_df       = pd.read_csv(os.path.join(data_dir, "test.csv"),  parse_dates=["date"])

    # Add date-based features to BOTH
    for df in [full_train_df, test_df]:
        df["year"]      = df["date"].dt.year
        df["month"]     = df["date"].dt.month
        df["day"]       = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek

    # Time-based split on the training CSV
    cutoff_date = "2017-01-01"
    mask_train  = full_train_df["date"] < cutoff_date

    train_df = full_train_df[mask_train].copy()      # training subset
    valid_df = full_train_df[~mask_train].copy()     # validation subset

def prep_data(train_df, cutoff_date = "2017-01-01", batch_size = 4096):
  input_features = ["year", "month", "day", "dayofweek", "store_nbr", "onpromotion", "family"]

  #convert features to be usable values (one-hot columns) ex: family values such as "dairy" or "deli" converted to their own collumns and given a value of 0 or 1
  X_all = pd.get_dummies(train_df[input_features], columns=["store_nbr", "family"], drop_first = 1)

  #featues
  feature_col = X_all.columns

  #sales (prediction target) log1p makes training more stable by reducing extreme spikes
  y_all = np.log1p(train_df["sales"].values)

  X_all = X_all.astype("float32")

  #split
  mask_train = train_df["date"] < cutoff_date

  X_train = X_all[mask_train].values
  X_valid = X_all[~mask_train].values

  y_train = y_all[mask_train]
  y_valid = y_all[~mask_train]

  #convert to tensors
  X_train_t = torch.tensor(X_train, dtype = torch.float32)
  X_valid_t = torch.tensor(X_valid, dtype = torch.float32)

  y_train_t = torch.tensor(y_train, dtype = torch.float32)
  y_valid_t = torch.tensor(y_valid, dtype = torch.float32)

  #data sets
  train_ds = TensorDataset(X_train_t, y_train_t)
  valid_ds = TensorDataset(X_valid_t, y_valid_t)

  #loaders
  train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
  valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = False)

  return train_dl, valid_dl, feature_col