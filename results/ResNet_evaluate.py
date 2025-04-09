import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from src.dataset import GTSRBLoadDataset
from models.resnet import ResNetBaseline
from src.config import NUM_CLASSES, BATCH_SIZE, DEVICE, NORMALIZATION_PARAMS, RESNET_CHECKPOINT_PATH, RESNET_FIGURE_PATH
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

    feature_extractor = torch.nn.Sequential(*list(model.resnet.children())[:-1])
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()
    
    all_preds = []
    all_labels = []
    all_embeddings = []
    all_image_paths = val_dataset.data

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            embedding = feature_extractor(images).view(images.size(0), -1)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_embeddings.append(embedding.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            progress_bar.set_postfix(val_loss=loss.item())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
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

    # Top missclassified
    misclassified_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cfm[i, j] > 0:
                misclassified_pairs.append((i, j, cfm[i, j]))
    misclassified_pairs = sorted(misclassified_pairs, key=lambda x: x[2], reverse=True)[:5]  
    print("\nTop 5 Misclassified Pairs (True -> Predicted, Count):")
    for true, pred, count in misclassified_pairs:
        print(f"Class {true} -> Class {pred}: {count} times")
        
    # Cosine Similarity
    similarity_matrix = cosine_similarity(all_embeddings)
    intra_class_sim = []
    inter_class_sim = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            if all_labels[i] == all_labels[j]:
                intra_class_sim.append(similarity_matrix[i, j])
            else:
                inter_class_sim.append(similarity_matrix[i, j])
    avg_intra_sim = np.mean(intra_class_sim) if intra_class_sim else 0.0
    avg_inter_sim = np.mean(inter_class_sim) if inter_class_sim else 0.0
    print(f"\nAverage Intra-Class Cosine Similarity: {avg_intra_sim:.4f}")
    print(f"Average Inter-Class Cosine Similarity: {avg_inter_sim:.4f}")
    
    # visualize embeddings with TNSE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings[:500])  
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels[:500], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Embeddings')
    tsne_png_path = os.path.join(RESNET_FIGURE_PATH, "resnet_tsne.png")
    plt.savefig(tsne_png_path)
    plt.close()
    print(f"t-SNE visualization saved as PNG to {tsne_png_path}")
    
    
    # Specific errors
    errors = [(path, true, pred) for path, true, pred in zip(all_image_paths, all_labels, all_preds) if true != pred]
    error_df = pd.DataFrame(errors, columns=['Image Path', 'True Label', 'Predicted Label'])
    error_csv_path = os.path.join(RESNET_FIGURE_PATH, "misclassified_samples.csv")
    error_df.to_csv(error_csv_path, index=False)
    print(f"\nMisclassified samples saved to {error_csv_path}")
    print(f"Number of misclassified samples: {len(errors)}")
    
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)])
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_resnet()