#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def load_metrics_grouped(metric_files_grouped):
    metrics_data = {}
    for metric_name, datasets in metric_files_grouped.items():
        metrics_data[metric_name] = {}
        for dataset_name, file_path in datasets.items():
            if file_path.exists():
                metrics_data[metric_name][dataset_name] = np.load(file_path)
            else:
                print(f"Warning: {file_path} does not exist.")
    return metrics_data


def plot_metrics(metrics_data, args):
    num_metrics = len(metrics_data)
    for metric_name, datasets in metrics_data.items():
        # For Dice metrics, we will create multiple subplots for per-class plots
        if metric_name.lower() == 'dice':
            # Determine number of classes from the data shape
            sample_dataset = next(iter(datasets.values()))
            if sample_dataset.ndim == 3:
                num_classes = sample_dataset.shape[2]
            else:
                num_classes = 1  # Default to 1 if data does not have classes

            fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))
            if num_classes == 1:
                axes = [axes]  # Ensure axes is iterable

            # Plot overall mean Dice
            ax = axes[0]
            E = None  # Number of epochs
            epochs = None

            # Initialize variables to store overall best metrics
            best_values = {}
            final_values = {}
            mean_values = {}
            std_values = {}

            for dataset_name, data in datasets.items():
                if E is None:
                    E = data.shape[0]
                    epochs = np.arange(1, E + 1)
                else:
                    if data.shape[0] != E:
                        print(f"Warning: {dataset_name} data for {metric_name} has different number of epochs.")

                if data.ndim == 3:
                    # Data shape is (epochs, samples, classes)
                    mean_per_epoch_per_class = data.mean(axis=1)  # Shape: (epochs, classes)
                    # Compute overall mean over classes
                    mean_per_epoch = mean_per_epoch_per_class.mean(axis=1)
                    ax.plot(epochs, mean_per_epoch, label=f"{dataset_name}")
                    best_epoch = np.argmax(mean_per_epoch) + 1
                    best_value = mean_per_epoch[best_epoch - 1]
                    final_value = mean_per_epoch[-1]
                    mean_value = mean_per_epoch.mean()
                    std_value = mean_per_epoch.std()

                    # Plot per-class Dice
                    class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']  # Adjust as needed
                    for class_idx in range(num_classes):
                        class_ax = axes[class_idx + 1]
                        class_mean = mean_per_epoch_per_class[:, class_idx]
                        class_label = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                        class_ax.plot(epochs, class_mean, label=f"{dataset_name}")
                        class_ax.set_title(f"{metric_name} - {class_label}")
                        class_ax.set_xlabel('Epoch')
                        class_ax.set_ylabel('Dice Coefficient')
                        class_ax.legend()
                        class_ax.grid(True)
                else:
                    raise ValueError(f"Unsupported data shape for {metric_name} - {dataset_name}: {data.shape}")

                # Store metrics for reporting
                best_values[dataset_name] = (best_value, best_epoch)
                final_values[dataset_name] = final_value
                mean_values[dataset_name] = mean_value
                std_values[dataset_name] = std_value

            # Configure overall Dice plot
            ax.set_title(f"{metric_name} - Overall Mean")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dice Coefficient')
            ax.legend()
            ax.grid(True)

            # Print reports after plotting both datasets
            for dataset_name in datasets.keys():
                best_value, best_epoch = best_values[dataset_name]
                final_value = final_values[dataset_name]
                mean_value = mean_values[dataset_name]
                std_value = std_values[dataset_name]

                print(f"\nReport for {metric_name} - {dataset_name}:")
                print("-" * 40)
                print(f"  Best Value: {best_value:.4f} at Epoch {best_epoch}")
                print(f"  Final Value: {final_value:.4f}")
                print(f"  Mean Value over Epochs: {mean_value:.4f}")
                print(f"  Std Dev over Epochs: {std_value:.4f}")

            fig.tight_layout()
            if args.dest:
                dest_file = args.dest.parent / f"{metric_name.lower()}_{args.dest.name}"
                fig.savefig(dest_file)
                print(f"Plot saved to {dest_file}")

            if not args.headless:
                plt.show()

        else:
            # For other metrics (e.g., Loss), plot training and validation together
            fig, ax = plt.subplots(figsize=(10, 5))
            E = None  # Number of epochs
            epochs = None

            # Initialize variables to store overall best metrics
            best_values = {}
            final_values = {}
            mean_values = {}
            std_values = {}

            for dataset_name, data in datasets.items():
                if E is None:
                    E = data.shape[0]
                    epochs = np.arange(1, E + 1)
                else:
                    if data.shape[0] != E:
                        print(f"Warning: {dataset_name} data for {metric_name} has different number of epochs.")

                if data.ndim == 1:
                    # Single value per epoch
                    ax.plot(epochs, data, label=f"{dataset_name}")
                    best_epoch = np.argmin(data) + 1 if 'loss' in metric_name.lower() else np.argmax(data) + 1
                    best_value = data[best_epoch - 1]
                    final_value = data[-1]
                    mean_value = data.mean()
                    std_value = data.std()
                elif data.ndim == 2:
                    # Multiple samples
                    mean_per_epoch = data.mean(axis=1)
                    ax.plot(epochs, mean_per_epoch, label=f"{dataset_name}")
                    best_epoch = np.argmin(mean_per_epoch) + 1 if 'loss' in metric_name.lower() else np.argmax(mean_per_epoch) + 1
                    best_value = mean_per_epoch[best_epoch - 1]
                    final_value = mean_per_epoch[-1]
                    mean_value = mean_per_epoch.mean()
                    std_value = mean_per_epoch.std()
                else:
                    raise ValueError(f"Unsupported data shape for {metric_name} - {dataset_name}: {data.shape}")

                # Store metrics for reporting
                best_values[dataset_name] = (best_value, best_epoch)
                final_values[dataset_name] = final_value
                mean_values[dataset_name] = mean_value
                std_values[dataset_name] = std_value

            ax.set_title(f"{metric_name}")
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True)

            # Print reports after plotting both datasets
            for dataset_name in datasets.keys():
                best_value, best_epoch = best_values[dataset_name]
                final_value = final_values[dataset_name]
                mean_value = mean_values[dataset_name]
                std_value = std_values[dataset_name]

                print(f"\nReport for {metric_name} - {dataset_name}:")
                print("-" * 40)
                print(f"  Best Value: {best_value:.4f} at Epoch {best_epoch}")
                print(f"  Final Value: {final_value:.4f}")
                print(f"  Mean Value over Epochs: {mean_value:.4f}")
                print(f"  Std Dev over Epochs: {std_value:.4f}")

            fig.tight_layout()
            if args.dest:
                dest_file = args.dest.parent / f"{metric_name.lower()}_{args.dest.name}"
                fig.savefig(dest_file)
                print(f"Plot saved to {dest_file}")

            if not args.headless:
                plt.show()


def run(args: argparse.Namespace) -> None:
    # Define metric files grouped by metric name
    metric_files_grouped = {
        'Loss': {
            'Training': args.metric_dir / 'loss_tra.npy',
            'Validation': args.metric_dir / 'loss_val.npy',
        },
        'Dice': {
            'Training': args.metric_dir / 'dice_tra.npy',
            'Validation': args.metric_dir / 'dice_val.npy',
        },
    }

    metrics_data = load_metrics_grouped(metric_files_grouped)

    if not metrics_data:
        print("No metrics data found to plot.")
        return

    plot_metrics(metrics_data, args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot and report metrics over time')
    parser.add_argument('--metric_dir', type=Path, required=True, metavar="METRIC_DIRECTORY",
                        help="The directory containing the metric .npy files.")
    parser.add_argument('--dest', type=Path, metavar="OUTPUT_PLOT.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and saves it directly (implies --dest to be provided).")

    args = parser.parse_args()

    if args.headless and not args.dest:
        parser.error("--headless requires --dest to be specified.")

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())