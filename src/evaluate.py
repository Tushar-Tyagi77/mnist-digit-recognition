import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load trained model
model = tf.keras.models.load_model("models/mnist_cnn.keras")

print("Model loaded successfully!")


# 2. Load MNIST test data
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# 3. Preprocess test data
X_test = X_test.astype("float32") / 255.0
X_test = X_test[..., tf.newaxis]


print("Test data shape:", X_test.shape)


# 4. Evaluate model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\nModel Evaluation")
print("-------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# 5. Generate predictions
predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(predictions, axis=1)


# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)


# 7. Classification Report
print("\nClassification Report")
print("-------------------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# 8. Plot Confusion Matrix
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("MNIST Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved successfully!")