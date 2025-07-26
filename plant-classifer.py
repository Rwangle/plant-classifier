import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflicts

import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_dir = "fruits train"
test_dir = "fruits test"

print("Training from:", train_dir)
print("Testing from:", test_dir)

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# Load Datasets
print("Loading datasets...")
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# Use smaller batch size for safer debugging
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Identity()
resnet18 = resnet18.to(device)
resnet18.eval()

# Feature Extraction Function
def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader):
            imgs = imgs.to(device)
            out = resnet18(imgs)
            features.append(out.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(features), np.array(labels)

print("Extracting training features...")
train_features, train_labels = extract_features(train_loader)
print("Done extracting training features.")

print("Extracting test features...")
test_features, test_labels = extract_features(test_loader)
print("Done extracting test features.")

print("Training classifiers...")

# SVM
svm = SVC(kernel='linear')
svm.fit(train_features, train_labels)
svm_preds = svm.predict(test_features)
svm_acc = accuracy_score(test_labels, svm_preds)

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_features, train_labels)
knn_preds = knn.predict(test_features)
knn_acc = accuracy_score(test_labels, knn_preds)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(train_features, train_labels)
logreg_preds = logreg.predict(test_features)
logreg_acc = accuracy_score(test_labels, logreg_preds)

print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"k-NN Accuracy: {knn_acc:.4f}")
print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")

print("Starting PCA visualization...")
pca = PCA(n_components=2)
reduced = pca.fit_transform(np.vstack([train_features, test_features]))
split = len(train_features)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced[split:, 0], reduced[split:, 1], c=test_labels, cmap="tab20", alpha=0.6)
plt.title("PCA of ResNet18 Test Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plot.png")
print("Plot saved as plot.png")
