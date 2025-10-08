import os
import numpy as np
import nibabel as nib
import torch
import torchcomplex as nn
from model_add import Model
from initializer import init_weights
from dataset import NiftiDataset
import matplotlib.pyplot as plt
import time


# -------------------------------------------------------
# Custom angle-based loss function
# -------------------------------------------------------
# This loss ensures that the predicted complex outputs
# have the correct *phase relationship* to the ground truth.
# We normalize both predicted and ground truth signals to
# unit magnitude (so only their phase remains), then compute
# squared error between them in the complex plane.
# -------------------------------------------------------
def angle_loss_fkt(preds, gts):
    pred = preds / (torch.abs(preds) + 1e-8)   # normalize predictions to unit circle
    gt = gts / (torch.abs(gts) + 1e-8)         # normalize ground truth
    diff = pred - gt                           # complex difference
    return torch.mean(diff.real**2 + diff.imag**2)  # mean squared error in real+imag parts


# -------------------------------------------------------
# Paths, device setup, and model initialization
# -------------------------------------------------------
this_file_path = "Models"
os.makedirs(this_file_path, exist_ok=True)  # create directory to store weights and logs
device = 'cuda:0'  # use GPU (first CUDA device)

# Initialize model architecture
model = Model().to(device)
model.apply(init_weights)   # apply custom initialization (e.g., Xavier/He normal)
'''
# (Optional) Load a previously saved model checkpoint
checkpoint_path = os.path.join(this_file_path, "Latest_Model.pt")
model.load_state_dict(torch.load(checkpoint_path))
'''
model.to(device)

# Optimizer (Adam with small learning rate)
# Adam is chosen for stability with complex-valued weights
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
)


# -------------------------------------------------------
# Dataset setup
# -------------------------------------------------------
# Custom dataset class loads NIfTI images (QSM/R2* maps),
# performs preprocessing, and yields forward simulations (fws)
# paired with corresponding ground-truth susceptibility maps (gts).
batch_size = 2
training_set = NiftiDataset(device=device, batch_size=batch_size)
validation_set = NiftiDataset(device=device, batch_size=batch_size)


# -------------------------------------------------------
# Loss tracking arrays for later visualization
# -------------------------------------------------------
mse_loss_fn = torch.nn.MSELoss()
training_losses = []
val_losses = []
training_mag_losses = []       # magnitude loss during training
training_angle_losses = []     # phase loss during training
validation_mag_losses = []     # magnitude loss during validation
validation_angle_losses = []   # phase loss during validation


# -------------------------------------------------------
# Utility: Gradient inspection
# -------------------------------------------------------
# Prints the gradient norm of each trainable parameter.
# Helps debug vanishing/exploding gradients and confirm
# that all layers receive meaningful updates.
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Layer {name} has gradient norm: {grad_norm:.4f}")


# -------------------------------------------------------
# Training loop (executed for one epoch)
# -------------------------------------------------------
# - Loops over mini-batches from the dataloader
# - Computes model predictions
# - Calculates two loss components:
#     (1) magnitude loss: compares |pred| vs |gt|
#     (2) angle/phase loss: compares arg(pred) vs arg(gt)
# - Combines them with weighting factors
# - Updates model parameters using backpropagation
# -------------------------------------------------------
def train_loop(dataloader, num_batches, accumulutation_steps, module_weight, phase_weight):
    model.train()  # enable training mode (dropout/batchnorm active)
    training_loss = 0
    total_mag_loss = 0
    total_angle_loss = 0

    for batch, (fws, gts) in enumerate(dataloader):
        preds = model(fws)  # forward pass: predict susceptibility maps from forward simulations

        # Weighted loss terms
        magnitude_loss = module_weight * mse_loss_fn(preds.abs(), gts.abs())
        angle_loss = phase_weight * angle_loss_fkt(preds, gts)

        # Final combined loss
        loss = magnitude_loss + angle_loss
        loss.backward()  # compute gradients

        # Gradient accumulation (if accumulutation_steps > 1)
        if (batch + 1) % accumulutation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Record batch losses
        training_loss += loss.item()
        total_mag_loss += magnitude_loss.item()
        total_angle_loss += angle_loss.item()

        # Print debug info every 10 batches
        if batch % 10 == 0:
            print(f'loss: {loss:>7f} | Mag Loss: {magnitude_loss:.6f} | Angle Loss: {angle_loss:.6f} [{batch}]')

        # Stop once num_batches reached
        if batch >= num_batches:
            break

    # Compute mean losses across all batches
    training_loss /= num_batches
    training_losses.append(training_loss)
    training_mag_losses.append(total_mag_loss / num_batches)
    training_angle_losses.append(total_angle_loss / num_batches)

    return training_loss, training_mag_losses[-1], training_angle_losses[-1]


# -------------------------------------------------------
# Validation loop (similar to training but no backprop)
# -------------------------------------------------------
# Used to monitor generalization on held-out data.
# Gradients are disabled for efficiency.
# -------------------------------------------------------
def validation_loop(dataloader, num_batches, module_weight, phase_weight):
    model.eval()  # set model to evaluation mode
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

    # Average over all validation batches
    validation_loss /= num_batches
    val_losses.append(validation_loss)
    validation_mag_losses.append(total_mag_loss / num_batches)
    validation_angle_losses.append(total_angle_loss / num_batches)

    return validation_loss, validation_mag_losses[-1], validation_angle_losses[-1]


# -------------------------------------------------------
# Dataloaders (stream mini-batches from dataset)
# -------------------------------------------------------
# Using batch_size=None because custom dataset already
# implements internal batching. This avoids double-batching.
train_loader = torch.utils.data.DataLoader(training_set, batch_size=None)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=None)


# -------------------------------------------------------
# Training orchestration (multi-epoch training)
# -------------------------------------------------------
best_epoch = 0
best_val_loss = float('inf')
best_model = None

accumulutation_steps = 1
num_epochs = 2000
total_start_time = time.time()
counter = 0

for t in range(0, num_epochs):
    # Dynamic weighting between magnitude and phase losses
    module_weight = 1e-3   # magnitude loss weight
    phase_weight = 1000    # phase loss weight (dominant)
    print(f'Epoch {t + 1}\n-------------------------------')

    # Run one epoch of training
    training_loss, avg_mag_loss, avg_angle_loss = train_loop(
        train_loader, 100, accumulutation_steps, module_weight, phase_weight
    )
    print(f"Training Loss: {training_loss:.6f} | Mag Loss: {avg_mag_loss:.6f} | Angle Loss: {avg_angle_loss:.6f}\n")

    # Run validation
    validation_loss, validation_magnitude_loss, validation_angle_loss = validation_loop(
        validation_loader, 5, module_weight, phase_weight
    )
    print(f"Validation Loss: {validation_loss:.6f} | Mag Loss: {validation_magnitude_loss:.6f} | Angle Loss: {validation_angle_loss:.6f}\n")

    # Track best model (lowest validation loss)
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_epoch = t + 1
        best_model = model.state_dict()

    # Periodically save intermediate weights (every 10 epochs)
    counter += 1
    if counter == 10:
        torch.save(model.state_dict(),
                   os.path.join(this_file_path, f"model_add.pt"))
        print(f"model_add_bs{batch_size}_acc{accumulutation_steps}_epoch[{t + 1}] saved at epoch {t + 1}")
        counter = 0

    # Save training/validation loss history after each epoch
    np.savez(
        os.path.join(this_file_path, "model_add_bs_loss_history.npz"),
        training_losses=training_losses,
        validation_losses=val_losses,
        training_mag_losses=training_mag_losses,
        training_angle_losses=training_angle_losses,
        validation_mag_losses=validation_mag_losses,
        validation_angle_losses=validation_angle_losses,
    )
    print("model_add_bs")


# -------------------------------------------------------
# Save the best-performing model across all epochs
# -------------------------------------------------------
torch.save(best_model, os.path.join(
    this_file_path,
    f"best_model_epoch_{best_epoch}.pt"
))
print(f"Best epoch: {best_epoch} with validation loss: {best_val_loss:.6f}")

# Final runtime report
total_training_time = time.time() - total_start_time
print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time / 60:.2f} minutes)")
