import tensorflow as tf
import numpy as np


# =========================
# 1. Load trained model
# =========================

model = tf.keras.models.load_model("models/mnist_cnn.keras")

print("Model loaded successfully!")


# =========================
# 2. Load MNIST test data
# =========================

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# =========================
# 3. Preprocess test images
# =========================

X_test = X_test.astype("float32") / 255.0

X_test = X_test[..., tf.newaxis]


# =========================
# 4. Select one image
# =========================

image = X_test[0]
actual_label = y_test[0]


# =========================
# 5. Make prediction
# =========================

prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)


# =========================
# 6. Get predicted digit
# =========================

predicted_digit = np.argmax(prediction[0])

confidence = np.max(prediction[0]) * 100


# =========================
# 7. Display result
# =========================

print("\nPrediction Result")
print("-------------------------")

print("Actual digit:", actual_label)
print("Predicted digit:", predicted_digit)
print(f"Confidence: {confidence:.2f}%")