# Disease Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict whether a breast tumor is malignant or benign based on medical diagnostic features.

Completed as part of the CodeAlpha Machine Learning Internship.

## Dataset
Breast Cancer Wisconsin Dataset (built into Scikit-learn).

- 569 records
- 30 input features
- 2 output classes: Malignant and Benign

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Models Used
1. Logistic Regression
2. Random Forest Classifier

## Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 98.25% | 95.61% |
| Precision | 98.61% | 95.89% |
| Recall | 98.61% | 97.22% |
| F1-Score | 98.61% | 96.55% |
| ROC-AUC | 99.54% | 99.37% |

## Best Model
Logistic Regression was selected based on its ROC-AUC score of 99.54%.

## Project Workflow
1. Loaded the dataset.
2. Explored the dataset.
3. Split data into training and testing sets.
4. Scaled numerical features.
5. Trained Logistic Regression and Random Forest models.
6. Evaluated model performance.
7. Generated a confusion matrix.
8. Saved the best model using Joblib.

## Saved Model
`disease_prediction_model.pkl`

## How to Run

Install dependencies:

```bash
pip install pandas matplotlib scikit-learn joblib
```

Run:

```bash
python disease_prediction.py
```

## Conclusion
The project demonstrates how machine learning classification algorithms can identify patterns in breast cancer diagnostic data. Logistic Regression achieved 98.25% accuracy on the test dataset.

**Note:** This is an educational machine learning project and is not intended for medical diagnosis.