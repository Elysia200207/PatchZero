import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, mode='adv0.5'):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of 'train', 'test', 'val'.
            num_classes (int): Total number of classes in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = 2
        self.mode = mode
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.cleans_dir = os.path.join(root_dir, split, 'cleans')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        self.labels_path = os.path.join(root_dir, split, 'labels', 'labels_.json')

        with open(self.labels_path, 'r') as file:
            self.labels = json.load(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_info = self.labels[idx]
        image_name = label_info['filename']
        clean_name = label_info['clean']
        mask_name = label_info['mask']
        
        if self.mode == 'clean':
            image_path = os.path.join(self.cleans_dir, clean_name)
            mask_path = None
        elif self.mode == 'adv':
            image_path = os.path.join(self.images_dir, image_name)
            mask_path = os.path.join(self.masks_dir, mask_name)
        else:      
            if random.random() > 0.5:
                image_path = os.path.join(self.images_dir, image_name)
                mask_path = os.path.join(self.masks_dir, mask_name)
            else:
                image_path = os.path.join(self.cleans_dir, clean_name)
                mask_path = None

        image = Image.open(image_path).convert('RGB')
        
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask[mask > 0] = 1  # 将非黑色像素值设为1
            mask = torch.tensor(mask, dtype=torch.int64)
            class_pre = torch.ones(self.num_classes, dtype=torch.float)
        else:
            mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.int64)
            class_pre = torch.zeros(self.num_classes, dtype=torch.float)
            class_pre[0]=1
            
        boxes = torch.tensor(label_info['boxes'], dtype=torch.float32)
        labels = torch.tensor(label_info['labels'], dtype=torch.int64)
        image_id = torch.tensor([label_info['image_id']], dtype=torch.int64)
        area = torch.tensor(label_info['area'], dtype=torch.float32)
        iscrowd = torch.tensor(label_info['iscrowd'], dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'masks': mask,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        if self.transform:
            image = self.transform(image)

        return image, mask, class_pre, targets
    
    def __len__(self):
        return len(self.labels)