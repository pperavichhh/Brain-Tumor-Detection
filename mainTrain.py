import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing import image
from PIL import Image
from keras.callbacks import TensorBoard, ReduceLROnPlateau

# Set the input size and image directory
INPUT_SIZE = 224  # VGG16 input size
image_directory = 'Dataset/'

# Load and preprocess your data
datasets = []
label = []

# Load your image data and create datasets and labels
no_tumor_image = os.listdir(image_directory + 'no/')
yes_tumor_image = os.listdir(image_directory + 'yes/')
glioma_tumor_image = os.listdir(image_directory + 'glioma_tumor/')
meningioma_tumor_image = os.listdir(image_directory + 'meningioma_tumor/')
pituitary_tumor_image = os.listdir(image_directory + 'pituitary_tumor/')

# print(no_tumor_image)

# path = 'no0.jpg'

# print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory +'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory +'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(1)

for i , image_name in enumerate(glioma_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory +'glioma_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(2)

for i , image_name in enumerate(meningioma_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory +'meningioma_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(3)

for i , image_name in enumerate(pituitary_tumor_image):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory +'pituitary_tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        datasets.append(np.array(image))
        label.append(4)

datasets = np.array(datasets)
label = np.array(label)



# Create train and test splits
x_train, x_test, y_train, y_test = train_test_split(datasets, label, test_size=0.2, random_state=0)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Load the VGG16 model with pre-trained weights (exclude top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

# Create a new model by adding your own classifier on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model with a reasonable initial learning rate
initial_learning_rate = 0.001  # You can adjust this value
optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define a learning rate scheduler
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Train the model
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True)
model.fit(x_train, y_train, epochs=25, batch_size=16, validation_data=(x_test, y_test), callbacks=[tensorboard_callback, reduce_lr_callback])

# Save the trained model
model.save('BrainTumorVGG16_25epochs.h5')
