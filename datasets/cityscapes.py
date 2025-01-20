import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', height=512, width=1024):
        super(CityscapesDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.height = height
        self.width = width
        
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_label = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64)))
        ])
        
        # Updated paths based on actual structure
        self.images_dir = os.path.join(root_dir, 'Cityscapes', 'Cityspaces', 'images', self.split)
        self.labels_dir = os.path.join(root_dir, 'Cityscapes', 'Cityspaces', 'gtFine', self.split)
        
        # Debug print
        print(f"Looking for images in: {self.images_dir}")
        print(f"Looking for labels in: {self.labels_dir}")
        
        self.images = []
        self.labels = []
        self._load_data()
        
    def _load_data(self):
        """Load data paths"""
        # First check if directories exist
        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise RuntimeError(f"Labels directory not found: {self.labels_dir}")
            
        # Get all cities
        cities = [d for d in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, d))]
        print(f"Found cities: {cities}")
        
        for city in cities:
            city_img_dir = os.path.join(self.images_dir, city)
            city_lbl_dir = os.path.join(self.labels_dir, city)
            
            if not os.path.exists(city_lbl_dir):
                print(f"Warning: No label directory for city {city}")
                continue
                
            # Get all images in city
            for filename in os.listdir(city_img_dir):
                if 'leftImg8bit.png' in filename:
                    img_path = os.path.join(city_img_dir, filename)
                    lbl_name = filename.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                    lbl_path = os.path.join(city_lbl_dir, lbl_name)
                    
                    if os.path.exists(lbl_path):
                        self.images.append(img_path)
                        self.labels.append(lbl_path)
                    else:
                        print(f"Warning: No label found for {filename}")
        
        if len(self.images) == 0:
            raise RuntimeError(f"No valid image-label pairs found in {self.images_dir}")
            
        print(f"Found {len(self.images)} images in {self.split} split")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        
        image = self.transform_image(image)
        label = self.transform_label(label)
        
        return image, label

    def get_class_names(self):
        return ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain',
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                'motorcycle', 'bicycle']