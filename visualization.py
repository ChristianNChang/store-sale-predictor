import numpy as np
import matplotlib.pyplot as plt

# Load the saved arrays
data = np.load("history.npz")

train_rmse = data["train_rmse"]
valid_rmse = data["valid_rmse"]
valid_rmsle = data["valid_rmsle"]

# Plot RMSE in log-space
plt.figure()
plt.plot(train_rmse, label="train_rmse")
plt.plot(valid_rmse, label="valid_rmse")
plt.xlabel("Epoch")
plt.ylabel("RMSE (log-space)")
plt.title("Training vs Validation RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("rmse_curve.png")
plt.show()
