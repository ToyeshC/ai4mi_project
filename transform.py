import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, shift, center_of_mass
from pathlib import Path
import shutil

def load_nifti(file_path):
    nii = nib.load(str(file_path))
    return nii.get_fdata(), nii.affine

def save_nifti(file_path, data, affine):
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(file_path))

def calculate_correction_matrix():
    rotation_correction = np.eye(4)

    # Translation correction (shift in z-direction)
    translation_correction = np.eye(4)
    translation_correction[:3, 3] = [0, 0, 10]  # shift: 10 units in z-direction

    # Combine rotations and translations
    correction_matrix = np.dot(translation_correction, rotation_correction)

    return correction_matrix

def compute_heart_shift(original_gt, corrected_gt):
    # Heart label is assumed to be '2'
    heart_original = (original_gt == 2).astype(np.float32)
    heart_corrected = (corrected_gt == 2).astype(np.float32)

    centroid_original = np.array(center_of_mass(heart_original))
    centroid_corrected = np.array(center_of_mass(heart_corrected))
    shift_vector = centroid_corrected - centroid_original

    return shift_vector

def apply_correction(image_data, correction_matrix, order=1):
    # Decompose correction_matrix into linear part and translation
    M = correction_matrix[:3, :3]
    t = correction_matrix[:3, 3]

    # Compute the inverse of the linear part
    M_inv = np.linalg.inv(M)

    # Compute the offset
    offset = -np.dot(M_inv, t)

    # Apply the affine transformation
    corrected_data = affine_transform(
        image_data,
        M_inv,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0
    )

    # Cast to original data type
    corrected_data = corrected_data.astype(image_data.dtype)
    return corrected_data

def shift_heart_segment(gt_data, shift_vector):
    # Heart label is assumed to be '2'
    heart_segment = (gt_data == 2).astype(np.float32)

    shifted_heart = shift(
        heart_segment,
        shift=shift_vector,
        order=0,       # Nearest-neighbor interpolation for labels
        mode='constant',
        cval=0.0
    )

    gt_data_corrected = np.where(gt_data == 2, 0, gt_data)  # Remove original heart label
    gt_data_corrected = np.where(shifted_heart > 0, 2, gt_data_corrected)  # Add shifted heart label

    # Cast to uint8
    return gt_data_corrected.astype(np.uint8)

def process_patient(patient_folder, correction_matrix, heart_shift_vector, output_folder):
    patient_output_folder = output_folder / patient_folder.name
    patient_output_folder.mkdir(parents=True, exist_ok=True)

    # Process CT image
    ct_path = patient_folder / f"{patient_folder.name}.nii.gz"
    ct_data, ct_affine = load_nifti(ct_path)
    corrected_ct = apply_correction(ct_data, correction_matrix, order=1)
    corrected_ct = np.round(corrected_ct).astype(ct_data.dtype)
    save_nifti(patient_output_folder / f"{patient_folder.name}.nii.gz", corrected_ct, ct_affine)

    # Process GT segmentation
    gt_path = patient_folder / "GT.nii.gz"
    gt_data, gt_affine = load_nifti(gt_path)
    corrected_gt = apply_correction(gt_data, correction_matrix, order=0)
    corrected_gt = shift_heart_segment(corrected_gt, heart_shift_vector)

    # Ensure labels are within valid range
    corrected_gt = np.clip(corrected_gt, 0, 4)
    save_nifti(patient_output_folder / "GT.nii.gz", corrected_gt, gt_affine)

def main():
    data_root = Path("/home/dev/ai4mi_project/data/segthor_train/train")
    output_root = Path("/home/dev/ai4mi_project/data/segthor_train_corrected/train")
    patient_27_folder = data_root / "Patient_27"

    output_root.mkdir(parents=True, exist_ok=True)

    # Load original and corrected GT for Patient 27
    original_gt, _ = load_nifti(patient_27_folder / "GT.nii.gz")
    corrected_gt, _ = load_nifti(patient_27_folder / "GT2.nii.gz")

    correction_matrix = calculate_correction_matrix()

    heart_shift_vector = compute_heart_shift(original_gt, corrected_gt)
    print(f"Heart shift vector: {heart_shift_vector}")

    for patient_folder in data_root.glob("Patient_*"):
        if patient_folder.name != "Patient_27":  # Skip patient 27 as it's already corrected
            print(f"Processing {patient_folder.name}...")
            process_patient(patient_folder, correction_matrix, heart_shift_vector, output_root)
        else:
            # Copy the already corrected Patient_27 data to the output folder
            patient_27_output = output_root / "Patient_27"
            patient_27_output.mkdir(parents=True, exist_ok=True)
            for file in patient_27_folder.glob("*"):
                if file.name.endswith(".nii.gz"):
                    shutil.copy(str(file), str(patient_27_output / file.name))

    print("All patients processed.")

if __name__ == "__main__":
    main()