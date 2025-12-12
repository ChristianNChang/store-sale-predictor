import torch
from torch import nn
from torch.utils.data import DataLoader
from data_prep import train_dl, device

class sales_model(nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(num_features, 256), #256 neuron hidden layer
        nn.ReLU(),
        nn.Linear(256, 128), #128 neuron hidden layer
        nn.ReLU(),
        nn.Linear(128,1) #out layer
    )

  #helper function to feed input and match 1D target
  def forward(self, x):
    return self.layers(x).squeeze(1)

#data was converted to log(1 + sales). This changes it back and returns RMSE
def to_RMSE(pred_log, true_log):
  pred = torch.expm1(pred_log)
  true = torch.expm1(true_log)
  mse = torch.mean((pred - true) ** 2)
  return torch.sqrt(mse)
