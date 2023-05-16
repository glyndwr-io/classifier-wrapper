from classifier import ClassifierWrapper, TrainingConfig, ImageClass, ImageDataset
from PIL import Image

# Before running the Demo, please download the following
# dataset and place it in ./datasets/demo/
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

# Open our test image
image1 = Image.open('./sample/daisy-1.jpg')

# Add our daisy image to the existing dataset
# Adding it twice shows that data won't be overwritten.
# This step and the wrapper.importDataset() step can
# be omitted since the training dataset is already present.
dataset = ImageDataset(
    name="demo",
    classes=[
        ImageClass(
            name="daisy",
            images=[image1, image1]
        )
    ]
)

# Set our training config
config = TrainingConfig(
    epochs=10,
    batchSize=32,
    trainingSplit=0.2,
    validationSplit=0.2,
    dropout=0.2
)

# Train and save a new model
wrapper = ClassifierWrapper()
wrapper.importDataset(dataset) # optional
wrapper.new('demo', config)
wrapper.save('demo')

# Make a prediction with out newly trained model
print(wrapper.predict(image1))
