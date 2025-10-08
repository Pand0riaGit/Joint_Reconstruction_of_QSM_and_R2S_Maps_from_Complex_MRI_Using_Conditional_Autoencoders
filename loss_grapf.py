import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load the saved losses ---
this_file_path = r"Models"  # adjust if needed
data = np.load(os.path.join(this_file_path, "model_no_add_bs_loss_history.npz"))

training_losses = data["training_losses"]
validation_losses = data["validation_losses"]
training_mag_losses = data["training_mag_losses"]
training_angle_losses = data["training_angle_losses"]
validation_mag_losses = data["validation_mag_losses"]
validation_angle_losses = data["validation_angle_losses"]

epochs = np.arange(1, len(training_losses) + 1)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Visualization Absolute, Magnitude and Angle loss for Training and Validation (Model No-Add)", fontsize=16)
marker_epoch = 1961

# Total loss
axes[0].plot(epochs, training_losses, label="Training", lw=2)
axes[0].plot(epochs, validation_losses, label="Validation", lw=2)
axes[0].axvline(marker_epoch, color="red", linestyle="--", lw=1.5, label=f"Epoch {marker_epoch}")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Magnitude loss
axes[1].plot(epochs, training_mag_losses, label="Training", lw=2)
axes[1].plot(epochs, validation_mag_losses, label="Validation", lw=2)
axes[1].axvline(marker_epoch, color="red", linestyle="--", lw=1.5, label=f"Epoch {marker_epoch}")
axes[1].set_title("Magnitude Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# Angle loss
axes[2].plot(epochs, training_angle_losses, label="Training", lw=2)
axes[2].plot(epochs, validation_angle_losses, label="Validation", lw=2)
axes[2].axvline(marker_epoch, color="red", linestyle="--", lw=1.5, label=f"Epoch {marker_epoch}")
axes[2].set_title("Angle Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
fig.savefig(r"Subplots\loss_no_add")
plt.show()
