import numpy as np

# Paths to your .npy files
loss_val_path = 'results/segthor/nnNet_hd/loss_val.npy'
dice_val_path = 'results/segthor/nnNet_hd/dice_val.npy'
# loss_val_path = 'results/segthor/enet_hd/loss_val.npy'
# dice_val_path = 'results/segthor/enet_hd/dice_val.npy'

# Load the metricsclear
loss_val = np.load(loss_val_path)
dice_val = np.load(dice_val_path)

# Handle incomplete or zero data
valid_epochs = []

for epoch in range(dice_val.shape[0]):
    epoch_dice = dice_val[epoch]
    dice_without_background = epoch_dice[:, 1:]  # Exclude background class
    mean_dice = np.mean(dice_without_background)
    if mean_dice > 0:
        valid_epochs.append(epoch)

if not valid_epochs:
    raise ValueError("No valid epochs found with non-zero Dice coefficient.")

# Find the best epoch based on mean Dice coefficient
best_epoch = None
best_mean_dice = -1

for epoch in valid_epochs:
    epoch_dice = dice_val[epoch]
    dice_without_background = epoch_dice[:, 1:]  # Exclude background class
    mean_dice = np.mean(dice_without_background)
    if mean_dice > best_mean_dice:
        best_mean_dice = mean_dice
        best_epoch = epoch

if best_epoch is None:
    raise ValueError("Could not find the best epoch.")

# Retrieve metrics from the best epoch
best_epoch_loss = loss_val[best_epoch]
best_epoch_dice = dice_val[best_epoch]

mean_loss = np.mean(best_epoch_loss)

dice_without_background = best_epoch_dice[:, 1:]
mean_dice_per_class = np.mean(dice_without_background, axis=0)
mean_dice = np.mean(dice_without_background)

# Print the best metrics
print(f"Best Epoch: {best_epoch}")
print(f"Validation Loss at Best Epoch: {mean_loss:.6f}")
print(f"Validation Mean Dice Coefficient at Best Epoch: {mean_dice:.6f}")
print("Dice Coefficient per Class:")
for idx, dice_score in enumerate(mean_dice_per_class, start=1):
    print(f"  Class {idx}: {dice_score:.6f}")
