import cv2 as cv
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.layers import TFSMLayer
from keras import Input, Model
from tensorflow.keras import datasets

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# CIFAR-10 class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plot first 16 training images
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i][0]])
plt.tight_layout()
plt.show()

# Reduce dataset size (optional for training, not needed here)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# Load SavedModel using TFSMLayer
print("Loading model...")
try:
    tfsm_layer = TFSMLayer("Image Classifier", call_endpoint="serving_default")  # <-- Make sure this is correct path
except Exception as e:
    print("Error loading model:", e)
    exit()

# Wrap in functional model
input_tensor = Input(shape=(32, 32, 3))
output_tensor = tfsm_layer(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)

print("Model loaded successfully.")

# Load and preprocess your input image
img = cv.imread('deer-5148320_640.jpg')
if img is None:
    print("Failed to load image. Check path.")
    exit()

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # CIFAR-10 input size
img = img / 255.0  # Normalize

# Show image
plt.imshow(img)
plt.axis('off')
plt.show()

# Run prediction
print("Running prediction...")
try:
    prediction = model.predict(np.array([img]), verbose=0)
    # TFSMLayer returns a dict, extract output
    prediction_array = prediction['output_0']

    # Top prediction
    index = np.argmax(prediction_array)
    print(f"Prediction: {class_names[index]}")

    # Optional: print all class probabilities
    print("\nClass probabilities:")
    for i, prob in enumerate(prediction_array[0]):
        print(f"{class_names[i]}: {prob:.4f}")

except Exception as e:
    print("Prediction failed:", e)
