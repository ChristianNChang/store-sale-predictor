import os
import numpy as np
import torch
from torch import nn
from torch import optim
import math

from data_prep import load_data, prep_data
from model_architecture import sales_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df, test_df = load_data(".")
train_dl, valid_dl, feature_col = prep_data(train_df)

batch, _ = next(iter(train_dl))
input_dim = batch.shape[1]

model = sales_model(input_dim).to(device)

def train_model(model, train_dl, valid_dl, epochs=20, lr=1e-3, weight_decay=1e-5,
                patience=5, clip_grad=None, print_every=1,):
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  criterion = nn.MSELoss()
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

  best_val = float('inf')
  best_state = None
  wait = 0

  hist = {'train_rmse': [], 'valid_rmse': [], 'valid_rmsle': []}

  for epoch in range(1, epochs+1):
    # training the model
    model.train()
    train_losses = []
    for xb, yb in train_dl:
      xb = xb.to(device)
      yb = yb.to(device) # log1p values
      optimizer.zero_grad()
      preds_log = model(xb) # predicded log1p(sales)
      loss = criterion(preds_log, yb) # MSE log space
      loss.backward()
      if clip_grad:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
      optimizer.step()
      train_losses.append(loss.item())

    train_rmse = math.sqrt(np.mean(train_losses))

    # validating the model
    model.eval()
    valid_losses = []
    valid_losses_rmsle = []
    with torch.no_grad():
      for xb, yb in valid_dl:
        xb = xb.to(device)
        yb = yb.to(device) # log1p values
        preds_log = model(xb) # predicted log1p(sales)
        loss = criterion(preds_log, yb) # MSE log space
        valid_losses.append(loss.item())
        valid_losses_rmsle.append(loss.item()) # same value RMSE on logs

    valid_rmse = math.sqrt(np.mean(valid_losses))
    valid_rmsle = math.sqrt(np.mean(valid_losses_rmsle))

    hist['train_rmse'].append(train_rmse)
    hist['valid_rmse'].append(valid_rmse)
    hist['valid_rmsle'].append(valid_rmsle)

    scheduler.step(valid_rmsle)

    if epoch % print_every == 0:
      print(f"Epoch {epoch:02d} | train_rmse (log-space) {train_rmse:.6f} | valid_rmse (log-space) {valid_rmse:.5f}")

    # for early stopping based on the valid_rmse
    if valid_rmse < best_val:
      best_val = valid_rmse
      best_state = model.state_dict()
      wait = 0
    else:
      wait += 1
      if wait == patience:
        print(f"Early stopping at epoch {epoch}")
        break

  if best_state is not None:
    model.load_state_dict(best_state)

  return model, hist

#run training
if __name__ == "__main__":

  # Get input dimension from batch
  batch, _ = next(iter(train_dl))
  input_dim = batch.shape[1]

  model = sales_model(input_dim).to(device)

  model, history = train_model(
      model,  train_dl, valid_dl,
      epochs = 50, lr = 1e-3, weight_decay = 1e-5,
      patience = 7, clip_grad = None, print_every = 1
  )