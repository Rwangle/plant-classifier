import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

#scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#hyperparameter grid for tuning
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.001]},
    {'kernel': ['poly'], 'C': [0.1, 1], 'gamma': ['scale'], 'degree': [2, 3]},
    {'kernel': ['sigmoid'], 'C': [0.1, 1], 'gamma': ['scale', 0.01]}
]

# Evaluate all combinations
results = []
for config in param_grid:
    keys, values = zip(*config.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        model = SVC(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            'kernel': params['kernel'],
            'params': params,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })


df = pd.DataFrame(results)

#Table with the comparisons between evaluation scores of the different SVM kernel types
#uses f1 score as the evaluation metric here, maybe accuracy is better?
best_per_kernel = df.loc[df.groupby('kernel')['f1_score'].idxmax()].reset_index(drop=True)
print("\n=== Best Configuration per Kernel ===")
print(best_per_kernel.to_string(index=False))

#Grouped bar chart for the different evaluation metric 
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
bar_data = best_per_kernel.set_index('kernel')[metrics]
ax = bar_data.plot(kind='bar', figsize=(10, 6))
plt.title('Evaluation Metrics for Best SVM Kernel Configurations')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Model with best f1 score
best_model_row = best_per_kernel.loc[best_per_kernel['f1_score'].idxmax()]
best_params = best_model_row['params']

#predict using the parameters for that model
best_model = SVC(**best_params)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Confusion matrix plot, as percentages instead of units, since there were inconsistent amounts of images per class
cm = confusion_matrix(y_test, y_pred_best)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
plt.title(f"Confusion Matrix - Best Model ({best_params['kernel']} Kernel)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()