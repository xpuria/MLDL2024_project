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
        
        # Define transforms
        self.transform_image = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_label = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            Lambda(lambda pic: torch.from_numpy(np.array(pic, np.int64)))
        ])
        
        # Setup paths based on your structure
        self.images_dir = os.path.join(self.root_dir, 'Cityscapes', 'Cityspaces', 'images', self.split)
        self.labels_dir = os.path.join(self.root_dir, 'Cityscapes', 'Cityspaces', 'gtFine', self.split)
        
        # Load file paths
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Load data paths based on directory structure"""
        # Get all cities in the split
        cities = sorted(os.listdir(self.images_dir))
        
        for city in cities:
            city_img_path = os.path.join(self.images_dir, city)
            city_lbl_path = os.path.join(self.labels_dir, city)
            
            if os.path.isdir(city_img_path) and os.path.isdir(city_lbl_path):
                # Get all images in the city
                for file_name in sorted(os.listdir(city_img_path)):
                    if file_name.endswith('leftImg8bit.png'):
                        image_path = os.path.join(city_img_path, file_name)
                        label_name = file_name.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                        label_path = os.path.join(city_lbl_path, label_name)
                        
                        if os.path.exists(label_path):
                            self.images.append(image_path)
                            self.labels.append(label_path)
        
        if len(self.images) == 0:
            raise RuntimeError(f'No valid images found in {self.images_dir}')
        
        print(f'Found {len(self.images)} images in {self.split} split')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            label = Image.open(self.labels[idx])
            
            # Apply transforms
            image = self.transform_image(image)
            label = self.transform_label(label)
            
            return image, label
        except Exception as e:
            print(f"Error loading image/label at index {idx}: {str(e)}")
            print(f"Image path: {self.images[idx]}")
            print(f"Label path: {self.labels[idx]}")
            raise e

    def get_class_names(self):
        """Get list of class names"""
        return [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]

    def verify_dataset(self):
        """Verify dataset paths and structure"""
        print("\nVerifying dataset structure...")
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
        
        if not os.path.exists(self.images_dir):
            print(f"Error: Images directory does not exist!")
            return False
            
        if not os.path.exists(self.labels_dir):
            print(f"Error: Labels directory does not exist!")
            return False
            
        print(f"\nFound {len(self.images)} images and {len(self.labels)} labels")
        print("Dataset verification complete!")
        return True