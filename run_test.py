#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import your model architecture
from ENet import ENet  # Adjust this if you are using a different model

# Import utility functions
from utils import probs2class, save_images

# Define dataset parameters
datasets_params = {
    "SEGTHOR_testset": {'K': 5, 'net': ENet, 'B': 32}  # Adjust batch size if necessary
}

class SliceDataset(Dataset):
    def __init__(self, mode, root_dir, img_transform=None, gt_transform=None, debug=False):
        self.mode = mode
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.debug = debug

        self.img_dir = self.root_dir / mode / 'img'
        print(f"[DEBUG] Looking for images in: {self.img_dir}")
        self.img_paths = sorted(self.img_dir.glob('*.png'))
        print(f"[DEBUG] Found {len(self.img_paths)} images in {self.img_dir}")

        # If ground truth exists, process it
        self.has_gt = False
        if self.gt_transform is not None:
            self.gt_dir = self.root_dir / mode / 'gt'
            if self.gt_dir.exists():
                self.gt_paths = sorted(self.gt_dir.glob('*.png'))
                print(f"[DEBUG] Found {len(self.gt_paths)} ground truth images in {self.gt_dir}")
                self.has_gt = True
            else:
                print(f"[DEBUG] Ground truth directory {self.gt_dir} does not exist.")

        if self.debug:
            self.img_paths = self.img_paths[:10]
            if self.has_gt:
                self.gt_paths = self.gt_paths[:10]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)

        sample = {'images': img, 'stems': img_path.stem}

        if self.has_gt:
            gt_path = self.gt_paths[idx]
            gt = Image.open(gt_path)
            if self.gt_transform:
                gt = self.gt_transform(gt)
            sample['gts'] = gt

        return sample

def setup_test(args):
    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f">> Using device: {device}")

    # Get dataset parameters
    dataset_key = 'SEGTHOR_testset'
    K = datasets_params[dataset_key]['K']
    net_class = datasets_params[dataset_key]['net']
    net = net_class(1, K)

    # Load the model
    if args.model_path.endswith('.pkl'):
        # Assuming you saved the entire model
        net = torch.load(args.model_path, map_location=device)
    else:
        # If you saved only the state dictionary
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    net.eval()

    B = datasets_params[dataset_key]['B']
    root_dir = Path(args.test_data_dir)

    # Image transformation
    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255.0,
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    # No ground truth transformation needed for test data
    gt_transform = None

    # Adjust the mode based on your directory structure
    test_set = SliceDataset('test',  # Adjust to '' if images are directly under root_dir/img/
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
    print(f">> Created test dataset with {len(test_set)} images...")

    test_loader = DataLoader(test_set,
                             batch_size=B,
                             num_workers=args.num_workers,
                             shuffle=False,
                             pin_memory=True)

    return net, device, test_loader, K

def runTest(args):
    print(f">>> Running test on SEGTHOR dataset")
    net, device, test_loader, K = setup_test(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tq_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc=">> Testing")
        for i, data in tq_iter:
            img = data['images'].to(device)
            stems = data['stems']

            # Get model predictions
            pred_logits = net(img)
            pred_probs = torch.nn.functional.softmax(pred_logits, dim=1)
            predicted_class = probs2class(pred_probs)
            mult = 63 if K == 5 else (255 / (K - 1))

            # Save the predictions
            save_images(predicted_class * mult,
                        stems,
                        output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model (bestmodel.pkl)")
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help="Directory containing the sliced test data")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the test predictions.")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--debug', action='store_true',
                        help="Use a small subset of the dataset for debugging.")

    args = parser.parse_args()
    runTest(args)

if __name__ == '__main__':
    main()
