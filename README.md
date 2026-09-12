# 🧠 MNIST Handwritten Digit Recognition

An end-to-end handwritten digit recognition system built using a Convolutional Neural Network (CNN), TensorFlow, FastAPI, HTML, CSS, and JavaScript.

The application allows users to draw a handwritten digit on a web canvas and uses a trained CNN model to predict the digit along with its confidence score.

---

## 🚀 Project Demo

The application provides an interactive web interface where users can:

- ✍️ Draw a handwritten digit
- 🔮 Get the predicted digit
- 📊 View prediction confidence
- 🧹 Clear the canvas and try again

---

## 📌 Features

- CNN-based handwritten digit classification
- MNIST dataset
- Image preprocessing
- Model training and evaluation
- Confusion matrix
- FastAPI REST API
- Interactive web-based drawing canvas
- Real-time digit prediction
- Prediction confidence score

---

## 🏗️ Project Architecture

```text
User
 │
 ▼
Web Canvas
 │
 ▼
JavaScript
 │
 │ POST /predict
 ▼
FastAPI
 │
 ▼
Image Preprocessing
 │
 ▼
CNN Model
 │
 ▼
Prediction
 │
 ▼
Digit + Confidence
 │
 ▼
Web Interface