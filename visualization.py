import numpy as np
import matplotlib.pyplot as plt

# Load the saved arrays
data = np.load("history.npz")

train_rmse = data["train_rmse"]
valid_rmse = data["valid_rmse"]
final_mape = data["final_mape"].item()

# Plot RMSE in log-space
plt.figure()
plt.plot(train_rmse, label="train_rmse")
plt.plot(valid_rmse, label="valid_rmse")
plt.xlabel("Epoch")
plt.ylabel("RMSE (log-space)")
plt.title("Training vs Validation RMSE")

plt.legend()

# Put final MAPE as an integer in the top-right corner
ax = plt.gca()
text_str = f"Final MAPE: {int(round(final_mape))}%"
ax.text(
    0.98, 0.98,
    text_str,
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plt.tight_layout()
plt.savefig("rmse_curve_with_mape.png")
plt.show()