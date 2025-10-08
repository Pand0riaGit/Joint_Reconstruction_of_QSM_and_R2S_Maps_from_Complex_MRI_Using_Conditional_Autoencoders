import os
import numpy as np
import nibabel as nib
import torch
import torchcomplex as nn
from model import Model
from initializer import init_weights
from dataset import NiftiDataset
import matplotlib.pyplot as plt
import time


# ======================================================
# Custom Loss Component: Angle Loss
# ======================================================
# This function measures the angular disagreement between
# complex predictions and ground truth, independent of
# their magnitude. It normalizes both tensors to unit
# modulus, then computes squared distance in the complex
# plane.
# ======================================================
def angle_loss_fkt(preds, gts):
    pred = preds / (torch.abs(preds) + 1e-8)
    gt = gts / (torch.abs(gts) + 1e-8)
    diff = pred - gt
    return torch.mean(diff.real**2 + diff.imag**2)


# ======================================================
# Experiment Setup
# ======================================================
this_file_path = "Models"
os.makedirs(this_file_path, exist_ok=True)
device = 'cuda:0'

# Instantiate model and initialize weights
model = Model().to(device)
model.apply(init_weights)

# Optimizer (Adam with standard parameters)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
)

# Training and validation datasets
batch_size = 2
training_set = NiftiDataset(device=device, batch_size=batch_size)
validation_set = NiftiDataset(device=device, batch_size=batch_size)

# Loss functions and trackers
mse_loss_fn = torch.nn.MSELoss()
training_losses, val_losses = [], []
training_mag_losses, training_angle_losses = [], []
validation_mag_losses, validation_angle_losses = [], []


# ======================================================
# Gradient Debugging (optional)
# ======================================================
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Layer {name} has gradient norm: {grad_norm:.4f}")


# ======================================================
# Training Loop
# ======================================================
# Computes loss, applies gradient updates, and logs
# magnitude + angle components separately.
# ======================================================
def train_loop(dataloader, num_batches, accum_steps, module_weight, phase_weight):
    model.train()
    total_loss, total_mag_loss, total_angle_loss = 0, 0, 0

    for batch, (fws, gts) in enumerate(dataloader):
        preds = model(fws)  # forward pass: predict susceptibility maps from forward simulations

        # Weighted loss terms
        magnitude_loss = module_weight * mse_loss_fn(preds.abs(), gts.abs())
        angle_loss = phase_weight * angle_loss_fkt(preds, gts)

        # Final combined loss
        loss = magnitude_loss + angle_loss
        loss.backward()

        # Gradient accumulation
        if (batch + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_mag_loss += magnitude_loss.item()
        total_angle_loss += angle_loss.item()

        if batch % 10 == 0:
            print(f'Batch {batch}: Loss={loss:.6f}, Mag={magnitude_loss:.6f}, Angle={angle_loss:.6f}')

        if batch >= num_batches:
            break

    avg_loss = total_loss / num_batches
    training_losses.append(avg_loss)
    training_mag_losses.append(total_mag_loss / num_batches)
    training_angle_losses.append(total_angle_loss / num_batches)

    return avg_loss, training_mag_losses[-1], training_angle_losses[-1]


# ======================================================
# Validation Loop
# ======================================================
# Same as training loop but without backpropagation.
# ======================================================
def validation_loop(dataloader, num_batches, module_weight, phase_weight):
    model.eval()
    validation_loss = 0
    total_mag_loss = 0
    total_angle_loss = 0

    with torch.no_grad():
        for batch, (fws, gts) in enumerate(dataloader):
            preds = model(fws)

            # Compute losses (same as training)

            angle_loss = phase_weight * angle_loss_fkt(preds, gts)
            magnitude_loss = module_weight * mse_loss_fn(preds.abs(), gts.abs())
            loss = magnitude_loss + angle_loss

            # Track losses
            validation_loss += loss.item()
            total_mag_loss += magnitude_loss.item()
            total_angle_loss += angle_loss.item()

            if batch >= num_batches:
                break

    avg_loss = validation_loss / num_batches
    val_losses.append(avg_loss)
    validation_mag_losses.append(total_mag_loss / num_batches)
    validation_angle_losses.append(total_angle_loss / num_batches)

    return avg_loss, validation_mag_losses[-1], validation_angle_losses[-1]


# ======================================================
# Dataloaders
# ======================================================
train_loader = torch.utils.data.DataLoader(training_set, batch_size=None)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=None)


# ======================================================
# Main Training Routine
# ======================================================
best_epoch, best_val_loss, best_model = 0, float('inf'), None
accum_steps = 1
num_epochs = 2000
total_start_time = time.time()
save_counter = 0

for epoch in range(num_epochs):
    module_weight, phase_weight = 1e-3, 1000
    print(f"\nEpoch {epoch + 1}\n-------------------------------")

    # Train one epoch
    train_loss, avg_mag_loss, avg_angle_loss = train_loop(train_loader, 100, accum_steps,
                                                          module_weight, phase_weight)
    print(f"Training: Loss={train_loss:.6f}, Mag={avg_mag_loss:.6f}, Angle={avg_angle_loss:.6f}")

    # Validate
    val_loss, val_mag_loss, val_angle_loss = validation_loop(val_loader, 5, module_weight, phase_weight)
    print(f"Validation: Loss={val_loss:.6f}, Mag={val_mag_loss:.6f}, Angle={val_angle_loss:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model = model.state_dict()

    # Periodic checkpointing
    save_counter += 1
    if save_counter == 10:
        torch.save(model.state_dict(),
                   os.path.join(this_file_path, f"model_no_add.pt"))
        print(f"Checkpoint saved at epoch {epoch + 1}")
        save_counter = 0

    # Save loss history after each epoch
    np.savez(
        os.path.join(this_file_path, "model_no_add_bs_loss_history.npz"),
        training_losses=training_losses,
        validation_losses=val_losses,
        training_mag_losses=training_mag_losses,
        training_angle_losses=training_angle_losses,
        validation_mag_losses=validation_mag_losses,
        validation_angle_losses=validation_angle_losses,
    )
    print("Loss history updated.")

# Save final best model
torch.save(best_model, os.path.join(this_file_path,
                                    f"best_model_epoch_{best_epoch}.pt"))
print(f"Best model at epoch {best_epoch} with validation loss {best_val_loss:.6f}")

# Timing report
total_training_time = time.time() - total_start_time
print(f"Total training time: {total_training_time:.2f} s ({total_training_time/60:.2f} min)")
