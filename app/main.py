from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import numpy as np

from app.preprocessing import preprocess_image


# Create FastAPI application
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API for handwritten digit recognition using CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = tf.keras.models.load_model("models/mnist_cnn.keras")


@app.get("/")
def home():
    return {
        "message": "MNIST Digit Recognition API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_data = await file.read()

    # Convert bytes to PIL Image
    image = Image.open(
        __import__("io").BytesIO(image_data)
    )

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(
        processed_image,
        verbose=0
    )

    # Get predicted digit
    predicted_digit = int(
        np.argmax(prediction[0])
    )

    # Get confidence
    confidence = float(
        np.max(prediction[0]) * 100
    )

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 2)
    }