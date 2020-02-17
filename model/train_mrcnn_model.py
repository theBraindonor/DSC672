

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import skimage

from skimage.io import imread
import pandas as pd

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from utility import use_project_path


class BuildingConfig(Config):
    """Configuration for training on the sample Building  dataset.
    Derives from the base Config class and overrides values specific
    to the building shapes dataset. See the CONFIG FILE for more attributes
    """
    NAME = "Building"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 background + 1 Building class
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    STEPS_PER_EPOCH = 100  # TODO: Find a clean way to override this.  It's just getting replaced in code for now
    VALIDATION_STEPS = 10 # TODO: Find a clean way to override this.  It's just getting replaced in code for now
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9]) # TODO: Update with better means


class BuildingDataset(utils.Dataset):
    def __init__(self, images_df):
        super().__init__()
        self.images_df = images_df

    def load_building(self, validation=False):
        """
        Load the buildings from the dataframe of images and the validation flag
        :param images_df:
        :param validation:
        :return:
        """
        self.add_class("building", 1, "building")
        for index, row in self.images_df.iterrows():
            if (validation and row['validation'] == 1.0) or (not validation and row['validation'] == 0.0):
                self.add_image("building", image_id=index, path=row['image'])

    def load_mask(self, image_id):
        """
        Return the mask for the selected image.
        :param image_id:
        :return:
        """
        # Grab the mask image using the pandas data frame and convert it into the needed format.
        image = self.images_df.iloc[image_id]
        mask = [imread(image['mask'])]
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.uint8)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


# Function used for display purposes
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def get_training_images_split(train_set, split):
    """
    Load a directory of training images and create a random split on them
    :param train_set:
    :param split:
    :return: A pandas dataframe with image, mask, and validation
    """

    # Quickly grab the file glob and stick it into a numpy array.  Done in numpy first, so that we can
    # rotate it before turning it into a data frame.
    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
        [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
    ]))
    training_df = pd.DataFrame(file_array, columns=['image', 'mask'])

    # Create an array of zeros and then set a random split fo them to 1, then save this as our validation column
    validation = np.zeros(len(training_df))
    indices = np.arange(len(training_df))
    random.shuffle(indices)
    for i in indices[:int(len(training_df)*split)]:
        validation[i] = 1.0
    training_df['validation'] = validation

    return training_df


if __name__ == '__main__':
    # Using the project path sets us to the root of the repository.
    use_project_path()

    # Temporary configuration variables will be contained here.  They will hopefully be updated to runtime
    # arguments once everything is working
    TRAINING_COLLECTION = 'sample_lg'
    VALIDATION_SIZE = 0.2
    BATCH_SIZE = 8
    INITIALIZE_WITH = 'coco'
    EPOCHS_HEAD = 1
    EPOCHS_RESNET = 1
    EPOCHS_ALL = 1

    MODEL_DIR = 'temp_data/logs'

    # This model will make use of the coco weights provided by the mask_rcnn project.
    COCO_MODEL_PATH = 'temp_data/mask_rcnn_coco.h5'
    if not os.path.exists(COCO_MODEL_PATH):
        print("Downloading Coco Weights...")
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Create a pandas array of training images.  Please note that this will perform a default 80/20 training
    # and testing split.
    training_images = get_training_images_split(TRAINING_COLLECTION, VALIDATION_SIZE)
    training_image_count = len(training_images[(training_images['validation'] == 0.0)])
    validation_image_count = len(training_images[(training_images['validation'] == 1.0)])

    # Create the mrcnn configuration object.  We are adjusting everything to understand batch size and using
    # the training and validation lengths as read back in from the pandas dataframe
    config = BuildingConfig()
    config.STEPS_PER_EPOCH = int(training_image_count / BATCH_SIZE)
    config.VALIDATION_STEPS = int(validation_image_count / BATCH_SIZE)
    config.display()

    # Load the training image dataset.
    dataset_train = BuildingDataset(training_images)
    dataset_train.load_building()
    dataset_train.prepare()

    # Load validation dataset, done so by setting validation to true
    dataset_val = BuildingDataset(training_images)
    dataset_val.load_building(validation=True)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    if INITIALIZE_WITH == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif INITIALIZE_WITH == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif INITIALIZE_WITH == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)



    #### Turn this on if you want to train the model
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training the head branches")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCHS_HEAD,
                layers='heads')


    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=(EPOCHS_HEAD + EPOCHS_RESNET),
                layers='4+')



    #### Turn this on if you want to train the model
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Training All Layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=(EPOCHS_HEAD + EPOCHS_RESNET + EPOCHS_ALL),
                layers="all")


    ###########################################
    # INFERENCE MODE
    inference_config = config

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # For now, we are just going to find the last weights file.
    model_path = model.find_last()[1]
    model.load_weights(model_path, by_name=True)

    #
    # Note for Sebastion: I have no idea how well the rest of this code works...I haven't gotten far enough to
    # test it out just yet.
    #

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)

    print("This is the image_ID", str(image_id)) #Find out which image is showing
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    #SHOW RANDOM IMAGE with MASK ON
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    # DETECT MASK FOR RANDOM IMAGE
    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], figsize =(8,8))
