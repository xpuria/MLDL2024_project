import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional

class CityscapesDataset(Dataset):
    """Cityscapes Dataset"""
    
    def __init__(self, root_path: str, split: str = 'train', 
                 img_size: Tuple[int, int] = (512, 1024), augment: bool = False):
        self.root = Path(root_path)
        self.split = split
        self.height, self.width = img_size  # Explicitly separate height and width
        self.augment = augment
        
        # Setup transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            transforms.PILToTensor()
        ])
        
        # Setup paths
        self.images_root = self.root / 'images' / split
        self.labels_root = self.root / 'gtFine' / split
        
        # Load data paths
        self.images = []
        self.labels = []
        self.cities = []
        self._load_data()
        
        # Print dataset info
        print(f"\nCityscapes {split} Dataset:")
        print(f"Number of samples: {len(self.images)}")
        print(f"Image size: {img_size}")
        print(f"Augmentations: {augment}")
        print(f"Cities ({len(set(self.cities))}): {', '.join(sorted(set(self.cities)))}")

    def _load_data(self):
        """Load all valid image-label pairs"""
        cities = [d for d in self.images_root.iterdir() if d.is_dir()]
        
        for city_path in sorted(cities):
            city_name = city_path.name
            
            for img_path in sorted(city_path.glob('*_leftImg8bit.png')):
                label_name = img_path.name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                label_path = self.labels_root / city_name / label_name
                
                if label_path.exists():
                    self.images.append(img_path)
                    self.labels.append(label_path)
                    self.cities.append(city_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        try:
            # Load image and label
            image = Image.open(self.images[idx]).convert('RGB')
            label = Image.open(self.labels[idx]).convert('L')
            
            # Apply transforms
            image = self.img_transform(image)
            label = self.label_transform(label).squeeze(0).long()  # Remove channel dim
            
            return image, label
            
        except Exception as e:
            print(f"Error loading sample {idx}:")
            print(f"Image: {self.images[idx]}")
            print(f"Label: {self.labels[idx]}")
            raise e

    def get_class_names(self):
        """Get list of class names"""
        return [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]