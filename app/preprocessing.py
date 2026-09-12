from PIL import Image
import numpy as np


def preprocess_image(image: Image.Image):
    # 1. Convert image to grayscale
    image = image.convert("L")

    # 2. Resize image to 28x28
    image = image.resize((28, 28))

    # 3. Convert image to NumPy array
    image = np.array(image)

    # 4. Normalize pixel values
    image = image.astype("float32") / 255.0

    # 5. Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # 6. Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image