import tkinter as tk
from tkinter import filedialog

import pandas as pd
from model_architecture import sales_model
from train import train_model
from data_prep import prep_data 

def run_training():
    train_path = filedialog.askopenfilename(title="Select train.csv")
    test_path  = filedialog.askopenfilename(title="Select test.csv")
    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df  = pd.read_csv(test_path, parse_dates=["date"])
    train_dl, valid_dl, feature_col = prep_data(train_df)
    model = sales_model(len(feature_col)).to(device)
    model, hist = train_model(model, train_dl, valid_dl)
    print("Training complete!")

root = tk.Tk()
root.title("Sales Forecasting GUI")

btn = tk.Button(root, text="Run Training", command=run_training)
btn.pack()

root.mainloop()