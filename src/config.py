import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../datasets/raw/GTSRB_Traffic_Sign')
CHECKPOINT_DIR = os.path.join(BASE_DIR, '../logs')
GTSRB_FIGURE_DIR = os.path.join(BASE_DIR, '../figures/GTSRB')

GTSRB_TRAIN_PATH = os.path.join(DATA_DIR, 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
GTSRB_TEST_PATH = os.path.join(DATA_DIR, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')
GTSRB_LABEL_PATH = os.path.join(BASE_DIR, '../data/label_name.json')
VGG16_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'vgg16.pth')
RESNET_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'resnet.pth')
RESNET_FIGURE_PATH = os.path.join(GTSRB_FIGURE_DIR, 'ResNet')

IMAGE_RESIZE = {
    'VGG16': (224, 224),
    'ResNet': (224, 224),
    # 'EfficientNetB7': (512, 512)
}

NORMALIZATION_PARAMS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PARAMS = {
    'patience': 3,
    'delta': 0.001,
    'verbose': True
}