import os
import numpy as np
import pandas as pd
from pathlib import Path

def load_metrics(result_dir, metric_name):
    metric_file = Path(result_dir) / f"{metric_name}.npy"
    return np.load(metric_file)

def get_best_epoch(result_dir):
    with open(Path(result_dir) / "best_epoch.txt", "r") as f:
        return int(f.read().strip())

def generate_tables(results_dir):
    models = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    # 1. Best Epoch Summary Table
    best_epoch_data = []
    for model in models:
        model_dir = os.path.join(results_dir, model)
        best_epoch = get_best_epoch(model_dir)
        val_dice = load_metrics(model_dir, "dice_val")
        best_dice = val_dice[best_epoch].mean()
        best_epoch_data.append({"Model": model, "Best Epoch": best_epoch, "Best Dice Score": best_dice})
    
    best_epoch_df = pd.DataFrame(best_epoch_data)
    print("Best Epoch Summary Table:")
    print(best_epoch_df.to_string(index=False))
    print("\n")

    # 2. Final Epoch Metrics Table
    final_epoch_data = []
    for model in models:
        model_dir = os.path.join(results_dir, model)
        val_dice = load_metrics(model_dir, "dice_val")
        val_loss = load_metrics(model_dir, "loss_val")
        final_epoch_data.append({
            "Model": model,
            "Final Dice Score": val_dice[-1].mean(),
            "Final Loss": val_loss[-1].mean()
        })
    
    final_epoch_df = pd.DataFrame(final_epoch_data)
    print("Final Epoch Metrics Table:")
    print(final_epoch_df.to_string(index=False))
    print("\n")

    # 3. Convergence Speed Table
    convergence_data = []
    for model in models:
        model_dir = os.path.join(results_dir, model)
        val_dice = load_metrics(model_dir, "dice_val")
        epochs_to_90_percent = np.argmax(val_dice.mean(axis=(1, 2)) > 0.9 * val_dice.max())
        convergence_data.append({
            "Model": model,
            "Epochs to 90% of max Dice": epochs_to_90_percent
        })
    
    convergence_df = pd.DataFrame(convergence_data)
    print("Convergence Speed Table:")
    print(convergence_df.to_string(index=False))
    print("\n")

    # 4. Class-wise Performance Table
    class_names = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]
    class_performance_data = []
    for model in models:
        model_dir = os.path.join(results_dir, model)
        val_dice = load_metrics(model_dir, "dice_val")
        best_epoch = get_best_epoch(model_dir)
        class_dice_scores = val_dice[best_epoch].mean(axis=0)
        class_performance_data.append({"Model": model, **{class_names[i]: score for i, score in enumerate(class_dice_scores)}})
    
    class_performance_df = pd.DataFrame(class_performance_data)
    print("Class-wise Performance Table (Best Epoch):")
    print(class_performance_df.to_string(index=False))

if __name__ == "__main__":
    results_dir = "results/segthor"  # Update this path to your results directory
    generate_tables(results_dir)
