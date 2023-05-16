
#Base Imports
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os

#TF and Keras imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#PyDantic Imports
from pydantic import BaseModel
from typing import List

# Config for training of a new model
class TrainingConfig(BaseModel):
    epochs: int | None = None
    batchSize: int | None = None
    mode: str | None = None
    dropout: float | None = None
    trainingSplit: float | None = None
    validationSplit: float | None = None

# Datatype for importing a Class of Images
class ImageClass(BaseModel):
    name: str
    images: List[PIL.Image.Image]

    class Config:
        arbitrary_types_allowed = True

# Datatype for importing a Dataset
class ImageDataset(BaseModel):
    name: str
    classes: List[ImageClass]


class ClassifierWrapper:
    def __init__(self):
        self.img_height = 180
        self.img_width = 180
        return
    
    def save(self, modelName: str):
        self.model.save(f'models/{modelName}')

    def load(self, modelName: str):
        try:
            # Load the model from file
            # And get the classnames from
            # the model's dataset
            self.model = keras.models.load_model(f'models/{modelName}')
            self.class_names = self.getClassNames(modelName)

            return True
        except Exception as e:
            print(e)
            return False

    def new(self, modelName: str, config: TrainingConfig):
        options = config.dict(exclude_none=True)
        mode = options.get('mode', 'classic')

        if mode == 'classic':
            self.train(modelName, options)
        # Any additional future training methods go here
        #elif mode == '':
        #   pass
        else:
            self.train(modelName, options)

        return

    def predict(self, image):
        # Resize the image to fit the model's dimensions
        image = image.resize((self.img_height, self.img_width))

        # Flatten the pixels into and array
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        # Make the prediction and store data
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Return the predicted class and confidence
        return {
            "prediction": self.class_names[np.argmax(score)],
            "confidence": 100 * np.max(score)
        }

    # --- Utility Methods ---
    def getClassNames(self, name: str):
        labels = []

        # Each folder in the dataset's directory represents
        # a class. If the dataset does not exist then we
        # return an empty array
        try:
            for root, dirs, files in os.walk(f'./datasets/{name}/'):
                label = root.split('/')[3]
                if(label != ''):
                    labels.append(label)
        except:
            pass

        return labels

    def hasDataset(self, name: str):
        return os.path.exists(f'./datasets/{name}/')

    def hasClass(self, dataset: str, classname: str):
        return os.path.exists(f'./datasets/{dataset}/{classname}/')

    def importDataset(self, dataset: ImageDataset):
        # if there is no directory for the dataset
        # we create it
        if(not self.hasDataset(dataset.name)):
            os.mkdir(f'./datasets/{dataset.name}/')
        
        for imageClass in dataset.classes:
            # if there is no sub-directory for the image class
            # we create it
            if(not self.hasClass(dataset.name, imageClass.name)):
                os.mkdir(f'./datasets/{dataset.name}/{imageClass.name}/')
            
            # Make sure to have a unique name to not overwrite existing
            # data
            i = 0
            for image in imageClass.images:
                temp = i
                while os.path.exists(f'./datasets/{dataset.name}/{imageClass.name}/image-{temp}.jpg'):
                    temp += 1           
                
                image.save(f'./datasets/{dataset.name}/{imageClass.name}/image-{temp}.jpg')       
        
        return

    # --- Training Functions ---
    # Implemented with guidance from the TensorFlow Documentation
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
    def train(self, modelName: str, options: dict):
        data_dir = pathlib.Path(f'datasets/{modelName}')

        # Initialize training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=options.get('trainingSplit',0.2),
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Initialize validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=options.get('validationSplit',0.2),
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=options.get('batchSize', 32)
        )

        # Store the class names from the directory
        # structure
        self.class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        # Cache our dataset
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Apply augmentation layers for variance
        # to prevent overfitting
        data_augmentation = keras.Sequential([
            layers.RandomFlip(
                "horizontal",
                input_shape=(
                    self.img_height,
                    self.img_width,
                    3
                )
            ),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Normalize layers for training
        self.model = Sequential([
            data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(options.get('dropout', 0.2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names))
        ])

        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model on our data
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=options.get('epochs', 10)
        )

        return history