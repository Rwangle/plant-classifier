# %%
import numpy as np

# Load pre-extracted features
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# %%
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Classes:", np.unique(y_train))

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize model
knn = KNeighborsClassifier()

# GridSearchCV
grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# %%
# Train model with grid search
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# %%
# Evaluate on test set
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Optional: Save the best model
import joblib
joblib.dump(best_knn, 'best_knn_model.pkl')

# Optional: Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(best_knn, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - k-NN")
plt.show()

# %%
