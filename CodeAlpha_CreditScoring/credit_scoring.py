from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Get the Credit Approval dataset
credit_data = fetch_ucirepo(id=27)

# Separate the input data and the target
X = credit_data.data.features
y = credit_data.data.targets
print("Dataset loaded successfully!")
print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())
print("\nDataset shape:")
print(X.shape)
print("\nMissing values in features:")
print(X.isnull().sum())
print("\nTarget values:")
print(y.value_counts())

# Handle missing values
for column in X.columns:
    if X[column].dtype.kind in "biufc":
        X[column] = X[column].fillna(X[column].median())
    else:
        X[column] = X[column].fillna(X[column].mode()[0])
print("\nMissing values after preprocessing:")
print(X.isnull().sum())

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(exclude=["number"]).columns
print("\nNumerical features:")
print(list(numeric_features))
print("\nCategorical features:")
print(list(categorical_features))

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
print("\nPreprocessing transformer created successfully!")

# Transform the features
X_transformed = preprocessor.fit_transform(X)
print("\nData transformed successfully!")
print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the Logistic Regression model
model = LogisticRegression(max_iter=5000)

# Train the model
model.fit(X_train, y_train.values.ravel())
print("\nModel trained successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test)
print("\nPredictions made successfully!")
print(y_pred[:10])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# Calculate other evaluation metrics
precision = precision_score(y_test, y_pred, pos_label="+")
recall = recall_score(y_test, y_pred, pos_label="+")
f1 = f1_score(y_test, y_pred, pos_label="+")

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Calculate ROC-AUC
class_index = list(model.classes_).index("+")
y_probability = model.predict_proba(X_test)[:, class_index]
y_test_binary = y_test["A16"].map({"+": 1, "-": 0})
roc_auc = roc_auc_score(y_test_binary, y_probability)
print("ROC-AUC:", roc_auc)

from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree
decision_tree.fit(X_train, y_train.values.ravel())
print("\nDecision Tree trained successfully!")

# Make predictions using the Decision Tree
dt_pred = decision_tree.predict(X_test)
print("\nDecision Tree predictions made successfully!")
print(dt_pred[:10])

# Calculate Decision Tree accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)
print("\nDecision Tree Accuracy:", dt_accuracy)
print("Decision Tree Accuracy (%):", dt_accuracy * 100)

# Calculate Decision Tree evaluation metrics
dt_precision = precision_score(y_test, dt_pred, pos_label="+")
dt_recall = recall_score(y_test, dt_pred, pos_label="+")
dt_f1 = f1_score(y_test, dt_pred, pos_label="+")
print("\nDecision Tree Precision:", dt_precision)
print("Decision Tree Recall:", dt_recall)
print("Decision Tree F1-Score:", dt_f1)

# Calculate Decision Tree ROC-AUC
dt_class_index = list(decision_tree.classes_).index("+")
dt_probability = decision_tree.predict_proba(X_test)[:, dt_class_index]
dt_roc_auc = roc_auc_score(
    y_test["A16"].map({"+": 1, "-": 0}),
    dt_probability
)
print("Decision Tree ROC-AUC:", dt_roc_auc)

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest.fit(X_train, y_train.values.ravel())
print("\nRandom Forest trained successfully!")
rf_pred = random_forest.predict(X_test)
print("Random Forest predictions made successfully!")
print(rf_pred[:10])

# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, pos_label="+")
rf_recall = recall_score(y_test, rf_pred, pos_label="+")
rf_f1 = f1_score(y_test, rf_pred, pos_label="+")
rf_class_index = list(random_forest.classes_).index("+")
rf_probability = random_forest.predict_proba(X_test)[:, rf_class_index]
rf_roc_auc = roc_auc_score(
    y_test["A16"].map({"+": 1, "-": 0}),
    rf_probability
)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Accuracy (%):", rf_accuracy * 100)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-Score:", rf_f1)
print("Random Forest ROC-AUC:", rf_roc_auc)

# Random Forest Feature Importance

feature_importances = random_forest.feature_importances_
feature_names = preprocessor.get_feature_names_out()
importance_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})
importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)
print("\nTop 10 Important Features:")
print(importance_data.head(10))

import matplotlib.pyplot as plt

# Select top 10 features
top_features = importance_data.head(10)
plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_pred, labels=["-", "+"])
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Not Approved (-)", "Approved (+)"]
)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

import joblib

# Save the trained Random Forest model and preprocessor
joblib.dump(random_forest, "random_forest_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("\nRandom Forest model saved successfully!")
print("Preprocessor saved successfully!")