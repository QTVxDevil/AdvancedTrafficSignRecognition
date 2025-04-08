import os
import json
import pandas as pd
from torch.utils.data import Dataset
from src.config import GTSRB_TRAIN_PATH, GTSRB_LABEL_PATH, IMAGE_RESIZE, NORMALIZATION_PARAMS
from sklearn.model_selection import train_test_split
import cv2
import torch

class GTSRBLoadDataset(Dataset):
    def __init__(self, transform=None, test_size=0.2, random_state=42, model_type='VGG16', mode='train'):
        self.transform = transform
        self.model_type = model_type
        self.image_size = IMAGE_RESIZE[model_type]
        self.data = []
        self.labels = []
        self.mode = mode
        self.label_mapping = self.load_label_mapping()
        self.load_data()
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )

    def load_label_mapping(self):
        with open(GTSRB_LABEL_PATH, 'r') as f:
            return json.load(f)

    def load_data(self):
        for label in os.listdir(GTSRB_TRAIN_PATH):
            label_path = os.path.join(GTSRB_TRAIN_PATH, label)
            csv_path = os.path.join(label_path, f"GT-{label}.csv")
            if os.path.isdir(label_path) and os.path.exists(csv_path):
                try:
                    annotations = pd.read_csv(csv_path, sep=';')
                    for _, row in annotations.iterrows():
                        img_path = os.path.join(label_path, row['Filename'])
                        self.data.append(img_path)
                        self.labels.append(row['ClassId'])
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

    def __getitem__(self, idx):
        data = self.train_data if self.mode == 'train' else self.val_data
        labels = self.train_labels if self.mode == 'train' else self.val_labels
        
        img_path = data[idx]
        label = labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image at {img_path} could not be read.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        
        if self.transform is None:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.train_data) if self.mode == 'train' else len(self.val_data)
