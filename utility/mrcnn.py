#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random

import numpy as np
import pandas as pd
from skimage.io import imread

from mrcnn.config import Config
from mrcnn import utils

from utility import split_mask_into_layers

#
#  These are common utilities the mask r-cnn model creation.  Basically a place to store common code.
#


class MaskRCNNBuildingConfig(Config):
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
    MEAN_PIXEL = np.array([129.79, 129.95, 108.03]) # TODO: Update with better means
    USE_MINI_MASK = False  # Giving this a try...
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # Giving this a try...

class MaskRCNNBuildingDataset(utils.Dataset):
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

    def get_image_filename(self, image_id):
        """
        Return the filename of a given image ID
        :param image_id:
        :return:
        """
        image = self.images_df.iloc[self.image_info[image_id]['id']]
        return image['image']

    def load_mask(self, image_id):
        """
        Return the mask for the selected image.
        :param image_id:
        :return:
        """
        # Grab the mask image using the pandas data frame and convert it into the needed format.
        image = self.images_df.iloc[self.image_info[image_id]['id']]
        mask = imread(image['mask'])

        # Split Mask Images before assigning ID to entire image
        mask_array = split_mask_into_layers(mask)

        # If run into an image that doesnt have building masks
        # Set the mask_array back to mask image
        if not mask_array:
            mask_array = [imread(image['mask'])]

        mask = np.stack(mask_array, axis=-1)
        result = mask, np.ones([mask.shape[-1]], dtype=np.uint8)
        return result

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def create_mrcnn_training_images_split(image_path, train_set, split):
    """
    Load a directory of training images and create a random split on them
    :param train_set:
    :param split:
    :return: A pandas dataframe with image, mask, and validation
    """

    # Quickly grab the file glob and stick it into a numpy array.  Done in numpy first, so that we can
    # rotate it before turning it into a data frame.
    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('%s/%s/tile-256/**/*.png' % (image_path, train_set), recursive=True)],
        [filename for filename in glob.iglob('%s/%s/mask-256/**/*.png' % (image_path, train_set), recursive=True)],
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
