from PIL import Image
import numpy as np


def preprocess_image(image: Image.Image):
    # Convert image to grayscale
    image = image.convert("L")

    # Convert to numpy array
    image = np.array(image)

    # Find the digit pixels
    # Canvas background = black (0)
    # Digit = white (> 30)
    coords = np.argwhere(image > 30)

    # If no digit is found
    if coords.size == 0:
        image = Image.fromarray(image)
        image = image.resize((28, 28))
        image = np.array(image).astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    # Find bounding box of the digit
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit
    cropped = image[y_min:y_max + 1, x_min:x_max + 1]

    # Convert cropped digit to PIL image
    cropped = Image.fromarray(cropped)

    # Keep some margin around the digit
    width, height = cropped.size
    size = max(width, height) + 10

    canvas = Image.new("L", (size, size), 0)

    # Center cropped digit inside square
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    canvas.paste(cropped, (x_offset, y_offset))

    # Resize to MNIST size
    canvas = canvas.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy
    image = np.array(canvas)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image