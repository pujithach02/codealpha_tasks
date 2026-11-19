
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

results = {}

# 4. Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    print("\n", name)
    for metric, score in results[name].items():
        print(f"{metric}: {score:.4f}")

# 5. Select model using ROC-AUC
best_name = max(results, key=lambda name: results[name]["ROC-AUC"])
best_model = models[best_name]
best_predictions = best_model.predict(X_test)

print("\nBest Model:", best_name)

# 6. Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)
display.plot()
plt.title("Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Save model
joblib.dump(best_model, "disease_prediction_model.pkl")

print("\nModel saved successfully!")
