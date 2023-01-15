import os
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
from utils import *
from preprocessing import *
import torchvision.transforms.functional as F


def check_mode(mode : str):
    assert mode.lower() in ['training', 'testing'], "Wrong value for data mode!"
    
    
MODE = ['Training', 'Testing']

class BrainTumorDataset(Dataset):
        
    def __init__(self, root_dir : str,
                 mode : str,
                 preprocess=False,
                 transforms=None,
                 augment_transforms=None):
        
       
        self.data_path = root_dir
        self.preprocess=preprocess
        self.transforms = transforms
        self.mode = mode
        self.augment_transforms=augment_transforms 
       
            
        self.images, self.labels = self.read_data()
        self.classes = self._get_classes()
        
      
    
    def apply_preprocessing(self, img):
        
        if not self.preprocess :
            return img
        
        img = crop_img(np.asarray(img))
        img = Image.fromarray(img)

        return img
        
            
    def set_transforms(self, new_transforms):
        self.transforms = new_transforms
                  
    
    
    def read_data(self):
        
        images, labels = [], []
        data_path = os.path.join(self.data_path, self.mode)
        if os.path.exists(data_path):
            paths = os.listdir(data_path)
            
        for label, p in enumerate(paths):
            path = os.path.join(data_path, p)
            if os.path.isdir(path):
                for image in os.listdir(path):
                    images.append(os.path.join(path, image))
                    labels.append(label)  
        return images, labels
    
    def indecis(self):
        indecis = torch.randperm(len(self.labels))
        return indecis
    
    
    def class_count(self):
        return torch.bincount(torch.tensor(self.labels))
    
    
    def class_weights(self, beta=0.999):
        num_per_class = self.class_count()
        eff_number = (1.0 - torch.pow(beta, num_per_class)) 
        weights = (1 - beta) / eff_number
        weights = weights / torch.sum(weights) * len(num_per_class)
        return weights
    
    
    def _get_classes(self):
        CLASSES = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
        return CLASSES
    
    
    def plot_data_distribution(self):
        x = self.classes
        y = np.bincount(self.labels)
        
        fig, ax = plt.subplots()
        bars = ax.barh(x, y)
        ax.bar_label(bars)
        plt.title("Class Distribution")
        plt.ylabel("Class Name")
        plt.xlabel("Class Count")
        plt.show()
            
        
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):
        
        image = Image.open(self.images[index]).convert('RGB')
        label = torch.as_tensor(self.labels[index])
        if self.augment_transforms:
            image = self.augment_transform(image)   
            
        image = self.apply_preprocessing(image)
        image = F.adjust_sharpness(img=image, sharpness_factor=2)
        image = F.adjust_contrast(img=image, contrast_factor=2) 
        image = F.adjust_gamma(img=image, gamma=0.9)

        
        if self.transforms:
            image = self.transforms(image)
        
            
        return image, label
        
  

class Subset(Dataset):
    
    def __init__(self, dataset, indices, transforms=None, augment_transforms=None):
        
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms
        self.augment_transforms=augment_transforms
        
        
    def __len__(self):
        return len(self.indices)
    
    def data_distribution(self):
        
        
        dist = np.array([0, 0, 0, 0])
        for i in self.indices:
            img, label =  self.dataset[i] 
            dist[label] +=1
        
        return dist
            
            
    def plot_data_distribution(self):
        x = self.dataset.classes
       
        y = self.data_distribution()
        
        fig, ax = plt.subplots()
        bars = ax.barh(x, y)
        ax.bar_label(bars)
        plt.title("Class Distribution")
        plt.ylabel("Class Name")
        plt.xlabel("Class Count")
        plt.show()
    
    
    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        if self.augment_transforms is not None:
            img = self.augment_transforms(img)
            
        return img, label
    
        
        
        
