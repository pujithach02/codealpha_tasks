import joblib

# Load saved model and preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

print("Saved model loaded successfully!")
print("Saved preprocessor loaded successfully!")