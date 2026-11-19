import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 2. Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# 6. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")

# 7. Predict handwritten digits
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Handwritten Digit Recognition - Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Display sample predictions
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.title(f"Predicted: {y_pred[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 10. Save model
model.save("handwritten_digit_model.keras")

print("\nModel saved successfully!")