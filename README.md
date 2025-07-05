# 🧠 CIFAR-10 Image Classifier with TensorFlow/Keras

This project implements an end-to-end **image classification system** using a **Convolutional Neural Network (CNN)** trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains 60,000 32×32 color images across 10 categories:

> 🛫 Plane, 🚗 Car, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚛 Truck

## 🚀 Features

- ✅ Trains a CNN using TensorFlow/Keras
- 🖼️ Visualizes training samples using `matplotlib`
- 💾 Saves and reloads the trained model using `TFSMLayer` (SavedModel format)
- 🧠 Loads and preprocesses real-world images using OpenCV
- 📊 Runs predictions and displays:
  - Top class prediction
  - Full class probability distribution

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

## 🧬 Model Architecture

```text
Input (32x32x3)
↓
Conv2D (32 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3) + ReLU
↓
Flatten
↓
Dense (64 units) + ReLU
↓
Dense (10 units) + Softmax
