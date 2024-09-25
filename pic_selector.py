import os
import random
import shutil

# Define the paths to your folders
source_image_folder = '/Users/toyesh/Documents/UvA/2024-25/P1/AI4MI/ai4mi_project/data/SEGTHOR/val/img'
source_groundtruth_folder = '/Users/toyesh/Documents/UvA/2024-25/P1/AI4MI/ai4mi_project/data/SEGTHOR/val/gt'
destination_image_folder = '/Users/toyesh/Documents/UvA/2024-25/P1/AI4MI/ai4mi_project/data/SEGTHOR mini/val/img'
destination_groundtruth_folder = '/Users/toyesh/Documents/UvA/2024-25/P1/AI4MI/ai4mi_project/data/SEGTHOR mini/val/gt'

# Create the destination folders if they don't exist
os.makedirs(destination_image_folder, exist_ok=True)
os.makedirs(destination_groundtruth_folder, exist_ok=True)

# List all files in the source image folder
all_images = os.listdir(source_image_folder)

# Filter to only include image files (e.g., jpg, png, etc.)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Adjust for your image types
images = [img for img in all_images if img.lower().endswith(image_extensions)]

# Randomly select 1000 images
selected_images = random.sample(images, 100)

# Copy the selected images and their ground truths
for image in selected_images:
    # Copy the image file
    shutil.copy(os.path.join(source_image_folder, image), os.path.join(destination_image_folder, image))
    
    # Copy the ground truth file (with the same name and extension)
    shutil.copy(os.path.join(source_groundtruth_folder, image), os.path.join(destination_groundtruth_folder, image))

print(f'Successfully copied {len(selected_images)} images and their corresponding ground truths.')
