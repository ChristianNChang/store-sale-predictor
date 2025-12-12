import os
import numpy as np
import torch
from torch import nn
from torch import optim
import math
import time

from data_prep import load_data, prep_data
from model_architecture import sales_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_df, test_df = load_data(".")
train_dl, valid_dl, feature_col = prep_data(train_df)

#get Mean Absolute Pecentage Error for sales units
def sales_units_MAPE(model, valid_dl, device, eps=1e-8):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for xb, yb_log in valid_dl:
            xb = xb.to(device)
            yb_log = yb_log.to(device)

            pred_log = model(xb)

            true_sales = torch.expm1(yb_log)
            pred_sales = torch.expm1(pred_log)

            all_true.append(true_sales.cpu().numpy())
            all_pred.append(pred_sales.cpu().numpy())

    y_true = np.concatenate(all_true).ravel()
    y_pred = np.concatenate(all_pred).ravel()

    # MAPE is undefined when y_true == 0, so skip those entries
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        print("Validation MAPE: no non-zero targets, returning NaN")
        return float("nan")

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    print(f"Validation MAPE (sales units): {mape:.2f}%  (ignored {(~mask).sum()} zero targets)")
    return mape

def train_model(model, train_dl, valid_dl, epochs=20, lr=1e-3, weight_decay=1e-5,
                patience=5, clip_grad=None, print_every=1,):
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  criterion = nn.MSELoss()
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,)

  best_val = float('inf')
  best_state = None
  wait = 0

  hist = {'train_rmse': [], 'valid_rmse': [], 'valid_rmsle': []}

  for epoch in range(1, epochs+1):

    #record start time
    start_time = time.perf_counter()
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

    #calculate epoch time
    epoch_time = time.perf_counter() - start_time

    if epoch % print_every == 0:
      print(f"Epoch {epoch:02d} | train_rmse (log-space) {train_rmse:.6f} | valid_rmse (log-space) {valid_rmse:.5f} | epoch_time {epoch_time:.2f}s")

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

  # Train the model and get training history (50 epochs)
  model, history = train_model(
      model,  train_dl, valid_dl,
      epochs = 50, lr = 1e-3, weight_decay = 1e-5,
      patience = 7, clip_grad = None, print_every = 1
  )

  # Save the trained model
  torch.save(model.state_dict(), "sales_model.pth")
  print("Saved trained model to sales_model.pth")

#get MAPE on sales units
mape = sales_units_MAPE(model, valid_dl, device)

# Save training history and MAPE to a .npz file
np.savez(
    "history.npz",
    train_rmse=np.array(history["train_rmse"]),
    valid_rmse=np.array(history["valid_rmse"]),
    valid_rmsle=np.array(history["valid_rmsle"]),
    final_mape=np.array(mape)
)
print("Saved training history and MAPE to history.npz")