import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', height=1024, width=512):
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
        
        # Setup directories
        self.images_dir = os.path.join(self.root_dir, 'images', self.split)
        self.labels_dir = os.path.join(self.root_dir, 'gtFine', self.split)
        
        # Load data paths
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # Get all city folders
        cities = os.listdir(self.images_dir)
        
        # Load data paths from each city
        for city in cities:
            city_img_path = os.path.join(self.images_dir, city)
            city_lbl_path = os.path.join(self.labels_dir, city)
            
            # Get all images in the city folder
            for file_name in os.listdir(city_img_path):
                if file_name.endswith('leftImg8bit.png'):
                    image_path = os.path.join(city_img_path, file_name)
                    label_name = file_name.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                    label_path = os.path.join(city_lbl_path, label_name)
                    
                    if os.path.exists(label_path):
                        self.images.append(image_path)
                        self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        
        # Apply transforms
        image = self.transform_image(image)
        label = self.transform_label(label)
        
        return image, label
