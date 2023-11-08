import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from cvzone.HandTrackingModule import HandDetector
from sklearn.model_selection import train_test_split
import numpy as np
import math
import time
import os

# Define image dimensions and batch size
imgSize = 150  # Reduce image size to save memory
batch_size = 32

# Function to preprocess and augment the data (resize, normalize, etc.)
def preprocess_data(image, label):
    image = tf.image.resize(image, (imgSize, imgSize))
    image = image / 255.0  # Normalize pixel values
    return image, label

# Load and preprocess image data using data generators
def load_and_preprocess_data(folder):
    gesture_classes = os.listdir(folder)
    x_train = []
    y_train = []

    for class_label, gesture_class in enumerate(gesture_classes):
        class_dir = os.path.join(folder, gesture_class)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            x_train.append(img)
            y_train.append(class_label)

    X_train, X_temp, y_train, y_temp = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create tf.data.Dataset for training data
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).map(preprocess_data).batch(batch_size)

    # Create tf.data.Dataset for validation data
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.map(preprocess_data).batch(batch_size)

    # Create tf.data.Dataset for test data
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.map(preprocess_data).batch(batch_size)

    return train_dataset, val_dataset, test_dataset

# Load data using data generators
train_dataset, val_dataset, test_dataset = load_and_preprocess_data('data')

# Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imgSize, imgSize, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the feature maps
model.add(layers.Flatten())

num_classes = len(os.listdir('data'))  # Automatically determine the number of classes

# Dense (fully connected) layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy}')
