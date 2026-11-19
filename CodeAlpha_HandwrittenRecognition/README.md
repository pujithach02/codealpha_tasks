# Handwritten Digit Recognition Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images.

Completed as part of the CodeAlpha Machine Learning Internship.

## Dataset
MNIST Handwritten Digit Dataset.

- 60,000 training images
- 10,000 testing images
- 28 × 28 pixel grayscale images
- 10 classes (digits 0–9)

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Model Architecture
The CNN model includes:
- Convolutional layers
- Max Pooling layers
- Flatten layer
- Dense layers
- Dropout layer
- Softmax output layer

## Project Workflow
1. Loaded the MNIST dataset.
2. Explored training and testing images.
3. Normalized image pixel values.
4. Built a CNN model.
5. Compiled the model using Adam optimizer.
6. Trained the model for 5 epochs.
7. Evaluated model performance.
8. Generated a confusion matrix.
9. Displayed sample digit predictions.
10. Saved the trained model.

## Model Performance

**Test Accuracy: 98.98%**

The model achieved approximately 99% accuracy on the test dataset.

## Saved Model
`handwritten_digit_model.keras`

## How to Run

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn joblib
```

Run:

```bash
python handwritten_recognition.py
```

## Conclusion
This project demonstrates how Convolutional Neural Networks can recognize handwritten digits with high accuracy using deep learning.