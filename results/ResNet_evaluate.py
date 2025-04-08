import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.dataset import GTSRBLoadDataset
from models.resnet import ResNetBaseline
from src.config import NUM_CLASSES, BATCH_SIZE, DEVICE, NORMALIZATION_PARAMS, RESNET_CHECKPOINT_PATH, RESNET_FIGURE_PATH
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_resnet():
    os.makedirs(RESNET_FIGURE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZATION_PARAMS['mean'], std=NORMALIZATION_PARAMS['std'])
    ])

    val_dataset = GTSRBLoadDataset(transform=transform, model_type='ResNet', mode='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ResNetBaseline(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(RESNET_CHECKPOINT_PATH))
    except FileNotFoundError:
        print(f"Error: Checkpoint file {RESNET_CHECKPOINT_PATH} not found.")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            progress_bar.set_postfix(val_loss=loss.item())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)])

    cfm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(NUM_CLASSES)], 
                yticklabels=[str(i) for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    cfm_png_path = os.path.join(RESNET_FIGURE_PATH, "resnet_confusion_matrix.png")
    plt.savefig(cfm_png_path)
    plt.close()
    print(f"Confusion matrix saved as PNG to {cfm_png_path}")

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_resnet()