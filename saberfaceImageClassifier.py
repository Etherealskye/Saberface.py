# This is the code used to train the classifier

import matplotlib as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import pathlib

load_dotenv()
# Path to save your model at 
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH_V2')
# Create checkpoint paths to save the model at 
checkpoint_path = CHECKPOINT_PATH
checkpoint_dir = os.path.dirname(checkpoint_path)


# Filepath of the dataset we will be using to train the classifier
TRAIN_PHOTOS = os.getenv('TRAIN_PHOTOS')
data_dir = pathlib.Path(TRAIN_PHOTOS)
# Load the images off the disk using keras.preprocessing.image_dataset_from_directory
# Define parameters for the loader

batch_size = 32
img_height = 180
img_width = 180

#We will use 80% for training and 20% for testing as we setup the dataset below
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed=123,
    image_size = (img_height, img_width),
    batch_size = batch_size 
)

#We can get the classnames from the dataset class_names attribute
class_names=train_ds.class_names
print(class_names)

# Make the dataset setup for performance
# We will use buffer prefetching (storing data in the cache) to avoid I/o blocking (So that the program does not need to wait for the file I/O operation to finish to continue)
# Two important methods are used:

# Datset.cache()
# Keeps the images in memory after they're loaded off the disk during the first epoch.
# This ensures that the dataset does not become a bottleneck
# Can also create a permanent on disk cache if the dataset is too large to fit into memory

# Datset.prefetch() overlaps data preprocessing and model exceution while training

AUTOTUNE=tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize the data
# RGB channel values are in the range from 0 to 255 - this is not ideal as we want values 
# to be between 0 and 1 for a neural networ
# We will do so with a rescaling layer.
def create_model(): 
    # Note that we do 1./255 to provide a float input indicating that we want to scale down values by 255   
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255,)
    # We will include this normalization layer inside our model definition

    # Crate the model - note that this model has not been tuned for accuracy
    # Aim for 80% accuracy with saberface mk-1

    #If left as is, the model will overfit and the accuracy will stall around 60% in the training process
    #We will  use data augmentation, L2 weight regularlization, and dropout to counter overfitting

    num_classes = 2

    # Data augmentation: When we generate additional training data from existing examples by augmenting 
    # them with random transformations to yield believable looking images
    # This helps expose the model to more aspects of the data and generalize better

    #Data_augmentation layers
    data_augmentation = keras.Sequential(
        [   
            layers.experimental.preprocessing.RandomFlip(
                "horizontal",
                input_shape = (img_height,img_width,3)
            ),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    # Dropout - Randomly drops out a number of output layer during the trianing process (sets it to zero)
    # Add dropout and data augmentation to our model
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # Randomly dropout 20% of the output units
        layers.Dropout(0.20),
        layers.Flatten(),    
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Compile the model
    model.compile(
        optimizer ='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    #model.summary()
    return model

if __name__ == "__main__":
    # Below code is for training the model (and so I don't stress my CPU everytime I Try to test the prediction accuracy)
    model = create_model()
    # model.load_weights(checkpoint_path)

    # Create a callback to save model to our specified checkpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
        )

    # Train dat boi & save it
    epochs = 50
    history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=epochs,
    callbacks=[cp_callback] # give the callback to training
    )




