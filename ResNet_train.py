import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.dataset import GTSRBLoadDataset
from models.resnet import ResNetBaseline
from src.config import NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NORMALIZATION_PARAMS, RESNET_CHECKPOINT_PATH, RESNET_FIGURE_PATH
from src.earlystopping import EarlyStopping
from src.config import EARLY_STOPPING_PARAMS

def train_resnet():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_PARAMS['mean'], std=NORMALIZATION_PARAMS['std'])
    ])

    train_dataset = GTSRBLoadDataset(transform=transform, model_type='ResNet', mode='train')
    val_dataset = GTSRBLoadDataset(transform=transform, model_type='ResNet', mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ResNetBaseline(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PARAMS['patience'],
        delta=EARLY_STOPPING_PARAMS['delta'],
        verbose=EARLY_STOPPING_PARAMS['verbose']
    )

    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {val_loss:.4f}')
        early_stopping(val_loss, model, RESNET_CHECKPOINT_PATH)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    print(f"Training complete. Model saved to: {RESNET_CHECKPOINT_PATH}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    loss_png_path = os.path.join(RESNET_FIGURE_PATH, "resnet_loss.png")
    plt.savefig(loss_png_path)
    plt.close()
    
if __name__ == "__main__":
    train_resnet()