import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from pathlib import Path
from affine import AFF, INV

def load_nifti(file_path, dtype=np.int16):
    """Load NIfTI image data and its affine, and cast the data to the specified type."""
    nii = nib.load(str(file_path))
    data = nii.get_fdata().astype(dtype)
    return data, nii.affine

def save_nifti(file_path, data, affine):
    """Save data as a NIfTI image."""
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(file_path))

def apply_affine_to_heart(gt_data, gt_affine, aff_matrix, inv_matrix):
    """Apply affine transformation only to the heart segment (label 2)."""
    heart_segment = (gt_data == 2).astype(gt_data.dtype)

    corrected_heart = affine_transform(
        heart_segment,
        inv_matrix[:3, :3],
        offset=inv_matrix[:3, 3],
        order=0,
        mode='constant',
        cval=0.0
    )

    gt_data_without_old_heart = np.where(gt_data == 2, 0, gt_data)
    corrected_gt = np.where(corrected_heart > 0.5, 2, gt_data_without_old_heart)

    return corrected_gt.astype(gt_data.dtype)

def adjust_affine(original_affine, aff_matrix):
    """Adjust the affine matrix of the image."""
    return np.dot(original_affine, aff_matrix)

def process_patient(patient_folder, aff_matrix, inv_matrix, output_folder):
    """Process the ground truth and CT image, applying the correction to the heart and adjusting affines."""
    patient_output_folder = output_folder / patient_folder.name
    patient_output_folder.mkdir(parents=True, exist_ok=True)

    # Process CT image
    ct_path = patient_folder / f"{patient_folder.name}.nii.gz"
    ct_data, ct_affine = load_nifti(ct_path, dtype=np.int16)
    adjusted_ct_affine = adjust_affine(ct_affine, aff_matrix)
    save_nifti(patient_output_folder / f"{patient_folder.name}.nii.gz", ct_data, adjusted_ct_affine)

    # Process GT segmentation
    gt_path = patient_folder / "GT.nii.gz"
    gt_data, gt_affine = load_nifti(gt_path, dtype=np.uint8)
    corrected_gt = apply_affine_to_heart(gt_data, gt_affine, aff_matrix, inv_matrix)
    adjusted_gt_affine = adjust_affine(gt_affine, aff_matrix)
    save_nifti(patient_output_folder / "GT.nii.gz", corrected_gt, adjusted_gt_affine)

def main():
    data_root = Path("/home/dev/ai4mi_project/data/segthor_train/train")
    output_root = Path("/home/dev/ai4mi_project/data/segthor_AFF/train")
    output_root.mkdir(parents=True, exist_ok=True)

    aff_matrix = AFF
    inv_matrix = INV
    print(f"Using AFF matrix:\n{aff_matrix}")
    print(f"Using INV matrix:\n{inv_matrix}")

    for patient_folder in data_root.glob("Patient_*"):
        print(f"Processing {patient_folder.name}...")
        process_patient(patient_folder, aff_matrix, inv_matrix, output_root)

    print("All patients processed.")

if __name__ == "__main__":
    main()