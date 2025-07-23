import os
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
from kagglehub import dataset_download

# Download dataset
dataset_path = dataset_download("omrathod2003/140-most-popular-crops-image-dataset")
data_dir = os.path.join(dataset_path, "RGB_224x224", "RGB_224x224", "train")

print("Using dataset from:", data_dir)

# -------------------------
# Step 1: Preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                         std=[0.229, 0.224, 0.225])    # ImageNet std
])

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

class_names = dataset.classes

# -------------------------
# Step 2: Feature Extraction (ResNet18)
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Identity()  # Remove classification head
resnet18 = resnet18.to(device)
resnet18.eval()

features = []
labels = []

print("Extracting features...")
with torch.no_grad():
    for imgs, lbls in tqdm(loader):
        imgs = imgs.to(device)
        out = resnet18(imgs)
        features.append(out.cpu().numpy())
        labels.extend(lbls.numpy())

features = np.vstack(features)
labels = np.array(labels)

print("Feature shape:", features.shape)  # (N, 512)

# -------------------------
# Step 3: Supervised Classifiers
# -------------------------

print("Training classifiers...")

# SVM
svm = SVC(kernel='linear')
svm.fit(features, labels)
svm_preds = svm.predict(features)
svm_acc = accuracy_score(labels, svm_preds)

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features, labels)
knn_preds = knn.predict(features)
knn_acc = accuracy_score(labels, knn_preds)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(features, labels)
logreg_preds = logreg.predict(features)
logreg_acc = accuracy_score(labels, logreg_preds)

print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"k-NN Accuracy: {knn_acc:.4f}")
print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")

# -------------------------
# Step 4: PCA + Clustering (Optional)
# -------------------------

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", alpha=0.6)
plt.title("PCA of ResNet18 Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
