import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional, List

class CityscapesDataset(Dataset):
    """Improved Cityscapes Dataset Implementation"""
    
    # Class-level constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]

    def __init__(
        self, 
        root_path: str, 
        split: str = 'train',
        img_size: Tuple[int, int] = (1024, 512),
        augment: bool = False
    ):
        self.root = Path(root_path)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # Path setup
        self._setup_paths()
        
        # Data loading
        self.samples = self._collect_samples()
        
        # Transform setup
        self._setup_transforms()
        
        # Dataset info
        self._log_dataset_info()

    def _setup_paths(self) -> None:
        """Setup dataset directory paths"""
        self.img_root = self.root / 'images' / self.split
        self.label_root = self.root / 'gtFine' / self.split
        
        if not self.img_root.exists():
            raise RuntimeError(f"Image path {self.img_root} does not exist")
        if not self.label_root.exists():
            raise RuntimeError(f"Label path {self.label_root} does not exist")

    def _collect_samples(self) -> List[Tuple[Path, Path, str]]:
        """Collect all valid image-label pairs and their cities"""
        samples = []
        cities = sorted([d for d in self.img_root.iterdir() if d.is_dir()])
        
        for city_path in cities:
            city_name = city_path.name
            
            for img_path in sorted(city_path.glob('*_leftImg8bit.png')):
                label_name = img_path.name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                label_path = self.label_root / city_name / label_name
                
                if label_path.exists():
                    samples.append((img_path, label_path, city_name))
        
        return samples

    def _setup_transforms(self) -> None:
        """Setup image and label transforms"""
        # Base transforms
        self.base_transforms = [
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ]
        
        # Optional augmentation transforms
        if self.augment:
            self.aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ]
            self.base_transforms = self.aug_transforms + self.base_transforms
        
        self.img_transform = transforms.Compose(self.base_transforms)
        
        self.label_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.NEAREST),
            transforms.PILToTensor()
        ])

    def _log_dataset_info(self) -> None:
        """Log dataset information"""
        print(f"\nCityscapes {self.split} Dataset:")
        print(f"Number of samples: {len(self.samples)}")
        print(f"Image size: {self.img_size}")
        print(f"Augmentations: {self.augment}")
        cities = sorted(set(city for _, _, city in self.samples))
        print(f"Cities ({len(cities)}): {', '.join(cities)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            img_path, label_path = self.image_paths[idx], self.label_paths[idx]
            
            # Load image and label
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            
            # Shape verification
            # Note: tensor shape is (C, H, W), but img_size is (H, W)
            expected_shape = (3, self.img_size[0], self.img_size[1])  # (C, H, W)
            if image.shape != expected_shape:
                print(f"Wrong image shape: got {image.shape}, expected {expected_shape}")
            if label.shape != (self.img_size[0], self.img_size[1]):
                print(f"Wrong label shape: got {label.shape}, expected {self.img_size}")
            
            return image, label
        except Exception as e:
            print(f"\nError loading sample {idx}:")
            print(f"Image path: {img_path}")
            print(f"Label path: {label_path}")
            raise e
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.CLASSES.copy()

    def get_class_distribution(self) -> dict:
        """Calculate class distribution in dataset"""
        dist = {name: 0 for name in self.CLASSES}
        
        for _, label_path, _ in self.samples:
            label = Image.open(label_path)
            label_array = np.array(label)
            unique, counts = np.unique(label_array, return_counts=True)
            
            for class_id, count in zip(unique, counts):
                if class_id < len(self.CLASSES):
                    dist[self.CLASSES[class_id]] += count
                    
        total = sum(dist.values())
        return {k: v/total for k, v in dist.items()}