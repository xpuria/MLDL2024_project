import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

class GTA5Dataset(Dataset):
    def __init__(self, root_dir, height=1024, width=512):
        super(GTA5Dataset, self).__init__()
        self.root_dir = root_dir
        self.height = height
        self.width = width
        
        # Define transforms for images only
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For labels, we'll handle the conversion manually
        self.transform_label = transforms.Resize((self.height, self.width), interpolation=Image.NEAREST)
        
        # Setup directories
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        
        # Color to train ID mapping
        self.color_to_id = {
            (128, 64, 128): 0,    # road
            (244, 35, 232): 1,    # sidewalk
            (70, 70, 70): 2,      # building
            (102, 102, 156): 3,   # wall
            (190, 153, 153): 4,   # fence
            (153, 153, 153): 5,   # pole
            (250, 170, 30): 6,    # traffic light
            (220, 220, 0): 7,     # traffic sign
            (107, 142, 35): 8,    # vegetation
            (152, 251, 152): 9,   # terrain
            (70, 130, 180): 10,   # sky
            (220, 20, 60): 11,    # person
            (255, 0, 0): 12,      # rider
            (0, 0, 142): 13,      # car
            (0, 0, 70): 14,       # truck
            (0, 60, 100): 15,     # bus
            (0, 80, 100): 16,     # train
            (0, 0, 230): 17,      # motorcycle
            (119, 11, 32): 18     # bicycle
        }
        
        # Load data paths
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # Get all images in the images directory
        for file_name in os.listdir(self.images_dir):
            if file_name.endswith('.png'):
                image_path = os.path.join(self.images_dir, file_name)
                label_path = os.path.join(self.labels_dir, file_name)
                
                if os.path.exists(label_path):
                    self.images.append(image_path)
                    self.labels.append(label_path)
        
        # Sort paths for deterministic behavior
        self.images.sort()
        self.labels.sort()

    def map_color_to_class(self, label_image):
        """Convert RGB label image to class indices"""
        label_array = np.array(label_image)
        h, w, _ = label_array.shape
        class_label = np.zeros((h, w), dtype=np.int64)
        
        # Create RGB tuple to class index mapping
        for color, class_idx in self.color_to_id.items():
            # Create mask for current color
            mask = np.all(label_array == color, axis=2)
            class_label[mask] = class_idx
        
        return torch.from_numpy(class_label)

    def get_class_names(self):
        """Return list of class names"""
        return [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle'
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('RGB')
        
        # Apply transforms
        image = self.transform_image(image)
        
        # For label, first resize then convert to class indices
        label = self.transform_label(label)  # This returns a PIL Image
        label = self.map_color_to_class(label)  # Convert to class indices
        
        return image, label