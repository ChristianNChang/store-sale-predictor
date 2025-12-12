import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_prep import load_data, prep_data
from model_architecture import sales_model

train_df, test_df = load_data(".")
train_dl, valid_dl, feature_cols = prep_data(train_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_test_loader(test_df, feature_cols, batch_size=8192):
    #set input features
    input_features = ["year", "month", "day", "dayofweek", "store_nbr", "onpromotion", "family"]

    #one hot encode test data
    X_test = pd.get_dummies(test_df[input_features], columns=["store_nbr", "family"], drop_first=True)

    #match training feature columns
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)
    X_test = X_test.astype("float32")

    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    test_ds = TensorDataset(X_test_t)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_dl

def main():
    #check device
    print("Using device:", device)

    # Load CSVs and build train loaders to get feature_cols
    train_df, test_df = load_data(".")
    train_dl, valid_dl, feature_cols = prep_data(train_df)

    input_dim = len(feature_cols)
    model = sales_model(num_features=input_dim)

    #load trained model weights
    model_path = "sales_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run train.py first to train and save the model."
        )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    model = model.to(device)
    model.eval()

    #build test data loader
    test_dl = build_test_loader(test_df, feature_cols)

    #make predictions
    preds_list = []
    with torch.no_grad():
        for (xb,) in test_dl:
            xb = xb.to(device)
            preds_log = model(xb)
            preds = torch.expm1(preds_log)
            preds_list.append(preds.cpu().numpy())
            
    all_preds = np.concatenate(preds_list)

    results_df = test_df.copy()
    results_df["predicted sales"] = all_preds

    out_path = "test_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(results_df[["predicted sales"]].head())


if __name__ == "__main__":
    main()