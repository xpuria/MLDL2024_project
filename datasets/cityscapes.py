import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda
from pathlib import Path

class CityscapesDataset(Dataset):
    """Enhanced Cityscapes Dataset class"""
    
    LABEL_SUFFIX = {
        'labelIds': 'gtFine_labelIds',
        'labelTrainIds': 'gtFine_labelTrainIds',
        'color': 'gtFine_color',
        'polygons': 'gtFine_polygons',
        'instanceIds': 'gtFine_instanceIds'
    }

    def __init__(self, root_dir, split='train', height=512, width=1024, 
                 label_type='labelTrainIds', transform=None, target_transform=None):
        """
        Args:
            root_dir: Path to Cityscapes dataset
            split: 'train', 'val', or 'test'
            height: Target height for resizing
            width: Target width for resizing
            label_type: Type of label to use
            transform: Optional transform for image
            target_transform: Optional transform for label
        """
        self.root = Path(root_dir)
        self.split = split
        self.height = height
        self.width = width
        self.label_suffix = self.LABEL_SUFFIX[label_type]
        
        # Default transforms if none provided
        self.transform = transform if transform is not None else self._default_transform()
        self.target_transform = target_transform if target_transform is not None else self._default_target_transform()
        
        # Setup paths based on actual structure
        self.images_root = self.root / 'cityscapes' / 'Cityscapes' / 'Cityspaces' / 'images' / split
        self.labels_root = self.root / 'cityscapes' / 'Cityscapes' / 'Cityspaces' / 'gtFine' / split
        
        print(f"Loading Cityscapes dataset from:")
        print(f"Images: {self.images_root}")
        print(f"Labels: {self.labels_root}")
        
        # Load image-label pairs
        self.image_paths, self.label_paths = self._load_data()
        
        print(f"Found {len(self.image_paths)} image-label pairs in {split} set")

    def _default_transform(self):
        """Default image transforms"""
        return transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _default_target_transform(self):
        """Default label transforms"""
        return transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x, np.int64)))
        ])

    def _load_data(self):
        """Load all image-label pairs"""
        image_paths = []
        label_paths = []
        
        # Get all city directories
        cities = [d for d in self.images_root.iterdir() if d.is_dir()]
        
        # For each city, get all images and corresponding labels
        for city_dir in sorted(cities):
            city_name = city_dir.name
            
            # Get all images in city
            for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
                # Construct corresponding label path
                label_name = img_path.name.replace('leftImg8bit', self.label_suffix)
                label_path = self.labels_root / city_name / label_name
                
                if label_path.exists():
                    image_paths.append(img_path)
                    label_paths.append(label_path)
                else:
                    print(f"Warning: Missing label for {img_path.name}")
        
        if not image_paths:
            raise RuntimeError(f"No valid image-label pairs found in {self.images_root}")
        
        return image_paths, label_paths

    def __len__(self):
        """Return total number of images"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get image and label at index"""
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Load label
            label_path = self.label_paths[idx]
            label = Image.open(label_path).convert('L')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image/label pair at index {idx}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Label path: {self.label_paths[idx]}")
            raise e

    def get_class_names(self):
        """Return list of class names"""
        return [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]

    def verify_dataset(self):
        """Verify dataset integrity"""
        print("\nVerifying dataset structure...")
        print(f"Total images found: {len(self.image_paths)}")
        print(f"Total labels found: {len(self.label_paths)}")
        
        # Check a few random samples
        indices = np.random.choice(len(self), min(5, len(self)), replace=False)
        for idx in indices:
            try:
                image, label = self[idx]
                print(f"\nSample {idx}:")
                print(f"Image shape: {image.shape}")
                print(f"Label shape: {label.shape}")
                print(f"Image range: ({image.min():.2f}, {image.max():.2f})")
                print(f"Unique label values: {torch.unique(label).numpy()}")
            except Exception as e:
                print(f"Error loading sample {idx}: {str(e)}")

def test_cityscapes_dataset(root_dir='/content/datasets', split='train'):
    """Test function for Cityscapes dataset"""
    dataset = CityscapesDataset(root_dir=root_dir, split=split)
    dataset.verify_dataset()
    
    # Create dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test batch loading
    images, labels = next(iter(loader))
    print("\nBatch test:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return images, labels