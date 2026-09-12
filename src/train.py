import tensorflow as tf
import matplotlib.pyplot as plt


# =========================
# 1. Load MNIST dataset
# =========================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# =========================
# 2. Normalize
# =========================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# =========================
# 3. Add channel dimension
# =========================

X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]


print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# =========================
# 4. Build CNN
# =========================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convert feature maps to vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# =========================
# 5. Display model
# =========================

model.summary()

# =========================
# 6. Compile the model
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# 7. Train the model
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# =========================
# 8. Evaluate on test data
# =========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)


# =========================
# 9. Save trained model
# =========================

model.save("models/mnist_cnn.keras")

print("\nModel saved successfully!")