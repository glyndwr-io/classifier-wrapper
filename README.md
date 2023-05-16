# Classifier Wrapper for TensorFlow

Classifier wrapper is designed to be an easy-to-use and extensible abstraction layer for creating, managing, and training Image Recognition models in TensorFlow. With your training data organized into classes, you can quickly and easily build a model that will predict which class a given image belongs to. The goal is to abstract away complex or boiler-plate code while giving access to essential training parameters.

## Dependencies

This wrapper requires tensorflow, PIL, pydantic and numpy

## Usage

Your interactions with Classifier Wrapper will happen mostly with PyDantic classes, both to make sure appropriate information is provided and provide clear documentation.

### Importing a Dataset

The First step is to load a dataset into Classifier Wrapper. You can provide a model name containing a list of classes, each containing a classname and a list of PIL images using the ImageDataset and ImageClass types. By using PIL image objects, you can perform any kind of data manipulation on the images before storing them for training. Once you have your ImageDataset object you can call the importDataset() method to store the dataset to disk. If your image data is already manipulated and organized, you can copy a folder of organized data to the datasets directory. The data organizational structure is as follows:

* model-name
  * class-1

    * image1.jpg
    * image2.jpg
    * ...
  * class-2

    * image1.jpg
    * image2.jpg
    * ...

If you already have copied a dataset or saved one to disk, you can still call importDataset() on an existing dataset name, and it will append images to existing classes or create new classes if they don't exist. This makes updating datasets with newly supplied data simple for retraining.

### Training a Dataset

Once the dataset has been imported, training can begin. Simply call the new() method with the dataset name. Calling the new() method with no configuration will provide a training environment with a default configuration, this should be used a baseline to then compare different configurations with to compare performance. Datasets will always be inherently different and there is no singular solution.

Once a model has been trained on a dataset, that model can be saved using the save() method. While the dataset and model should have the same name, you may wish to save variations to compare performance (ex. training a dataset flowers and saving flowers-1, flowers-2 etc).

### Making a Prediction from a Trained Model

To make a prediction on a trained model, the model first has to be loaded. If you have just trained a model, then the loading step is not neccesary. Otherwise, calling the load() method with the model name will load the model into memory, and return a boolean based on whether or not it was loaded successfully. Once the model is loaded, you can provide the predict() method with an image and it will return a dictionary providing you with the model's prediction and confidence.

## Examples

You can find an example in demo.py. Before running the demo, please download the following dataset, extract it, and place the classes in ./datasets/demo/
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
