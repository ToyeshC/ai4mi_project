import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import torch
from glob import glob
from pathlib import Path
from typing import Dict
import re

from utils import dice_coef, class2one_hot

def extract_files(pattern: str, id_pattern: str) -> Dict[str, sitk.Image]:
    files = glob(pattern, recursive=True)
    id_regex = re.compile(id_pattern)
    results = {}
    for file in files:
        match = id_regex.search(file)
        if match:
            patient_id = match.group(1)
            results[patient_id] = sitk.ReadImage(file)
    return results

def normalize_labels(image: torch.Tensor, num_classes: int):
    """
    Normalize labels to be in the range [0, num_classes - 1].
    This function assumes that the input image might contain values in a larger range, 
    and we need to scale them down to [0, num_classes - 1].
    """
    max_value = torch.max(image)
    if max_value > num_classes - 1:
        # print(f"Warning: Normalizing labels with max value {max_value}")
        image = (image.float() * (num_classes - 1) / max_value).long()
    return image

def evaluate_segmentation(ground_truth: sitk.Image, prediction: sitk.Image, num_classes: int):
    # align images
    if ground_truth.GetOrigin() != prediction.GetOrigin():
        prediction.SetOrigin(ground_truth.GetOrigin())
    if ground_truth.GetSpacing() != prediction.GetSpacing():
        prediction.SetSpacing(ground_truth.GetSpacing())
    if ground_truth.GetDirection() != prediction.GetDirection():
        prediction.SetDirection(ground_truth.GetDirection())
    
    prediction = sitk.Resample(prediction, ground_truth) # match ground truth

    gt_np = sitk.GetArrayFromImage(ground_truth)
    pred_np = sitk.GetArrayFromImage(prediction)

    dice_scores = []
    for gt_slice, pred_slice in zip(gt_np, pred_np):
        gt_tensor = torch.tensor(gt_slice, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        pred_tensor = torch.tensor(pred_slice, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        
        # Normalize labels
        gt_tensor = normalize_labels(gt_tensor, num_classes)
        pred_tensor = normalize_labels(pred_tensor, num_classes)
       
        # Convert to one-hot encoding
        gt_one_hot = class2one_hot(gt_tensor, num_classes)
        pred_one_hot = class2one_hot(pred_tensor, num_classes)

        # Calculate Dice coefficient
        dice_score = dice_coef(gt_one_hot, pred_one_hot)
        dice_scores.append(dice_score.mean().item())

    mean_dice = np.mean(dice_scores)
    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(ground_truth, prediction)
    hausdorff_dist = hausdorff_filter.GetHausdorffDistance()
    avg_hausdorff_dist = hausdorff_filter.GetAverageHausdorffDistance()
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(ground_truth, prediction)
    # jaccard_coeff = overlap_filter.GetJaccardCoefficient()
    vol_similarity = overlap_filter.GetVolumeSimilarity()
    false_negative = overlap_filter.GetFalseNegativeError()
    false_positive = overlap_filter.GetFalsePositiveError()

    return {
        'dice': mean_dice,
        'hausdorff': hausdorff_dist,
        'avg_hausdorff': avg_hausdorff_dist,
        # 'jaccard': jaccard_coeff,
        'vol_similarity': vol_similarity,
        'false_negative': false_negative,
        'false_positive': false_positive
    }

def plot_results(results: Dict[str, Dict[str, float]], save_path: Path):
    metrics = list(next(iter(results.values())).keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5*num_metrics))
    
    for i, metric in enumerate(metrics):
        ids = list(results.keys())
        values = [results[id_][metric] for id_ in ids]
        
        axes[i].bar(ids, values)
        axes[i].set_title(f'{metric.capitalize()} per Image')
        axes[i].set_xlabel('Image ID')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure saved as {save_path}")
    plt.close()

def save_results_to_csv(results: Dict[str, Dict[str, float]], save_path: Path):
    metrics = list(next(iter(results.values())).keys())
    
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID'] + metrics)
        
        for patient_id, patient_metrics in results.items():
            writer.writerow([patient_id] + [patient_metrics[metric] for metric in metrics])
    
    print(f"Results saved to {save_path}")

def run(args):
    Path(args.dest_folder).mkdir(parents=True, exist_ok=True)
    ground_truth_pattern = os.path.join(args.source_scan_pattern, "Patient_*/GT.nii.gz")
    ground_truths = extract_files(ground_truth_pattern, r"Patient_(\d+)")
    print(f"Found {len(ground_truths)} ground truth files.")

    prediction_pattern = os.path.join(args.prediction_folder, "Patient_*.nii.gz")
    predictions = extract_files(prediction_pattern, r"Patient_(\d+)")
    print(f"Found {len(predictions)} prediction files.")

    results = {}
    num_classes = 5
    
    for patient_id, pred_image in predictions.items():
        if patient_id in ground_truths:
            gt_image = ground_truths[patient_id]
            metrics = evaluate_segmentation(gt_image, pred_image, num_classes)
            results[patient_id] = metrics
            print(f"ID: {patient_id}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"Warning: No ground truth file found for ID {patient_id}")

    if not results:
        print("No results to process. Please check your file paths and patterns.")
        return

    plot_results(results, Path(args.dest_folder) / 'metrics.png')
    save_results_to_csv(results, Path(args.dest_folder) / 'metrics.csv')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help='Pattern for ground truth scans, e.g., "/home/dev/ai4mi_project/data/segthor_train/train"')
    parser.add_argument('--prediction_folder', type=str, required=True,
                        help='Path to the folder containing prediction files, e.g., "/home/dev/ai4mi_project/volumes/segthor/enet_test"')
    parser.add_argument('--dest_folder', type=str, required=True,
                        help='Path to the folder where results will be saved, e.g., "results/plots"')
    return parser.parse_args()

if __name__ == "__main__":
    run(get_args())
