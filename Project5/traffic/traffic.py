import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Loop through each folder (categories 0 to NUM_CATEGORIES - 1)
    for category in range(NUM_CATEGORIES):  
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue  # Skip if the folder doesn't exist

        # Loop through all image files in the category folder
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            try:
                # Read image
                image = cv2.imread(file_path)

                # Resize image to the desired dimensions
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

                # Append the image and corresponding label
                images.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Convert to numpy arrays for further processing
    return np.array(images), np.array(labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Define the model
    model = Sequential([
        # Convolutional Layer
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        BatchNormalization(),
        
        # (Optional) Pooling Layer
        MaxPooling2D(pool_size=(3, 3)),
        
        # Flatten the output
        Flatten(),
        
        # Fully connected hidden layer
        Dense(128, activation="relu"),
        BatchNormalization(),
        
        # Output layer for classification
        Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Summarize the model structure
    model.summary()

    return model


if __name__ == "__main__":
    main()
