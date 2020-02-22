

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
import matplotlib.image as mpimg
import glob
import skimage
from scipy import ndimage
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
        image = self.images_df.iloc[self.image_info[image_id]['id']]
        mask = imread(image['mask'])

        # Split Mask Images before assigning ID to entire image
        mask_array = create_separate_mask(mask)

        # If run into an image that doesnt have building masks
        # Set the mask_array back to mask image
        if not mask_array:
            mask_array = [imread(image['mask'])]

        mask = np.stack(mask_array, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.uint8)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Function used to Split the mask Images
def create_separate_mask(mask):
    # bring in mask
    label_im, nb_labels = ndimage.label(mask)
    # Set up empty array to hold split mask images
    mask_array = []
    for i in range(nb_labels):
        # create an array which size is same as the mask but filled with
        # values that we get from the label_im.
        # If there are three masks, then the pixels are labeled
        # as 1, 2 and 3.

        mask_compare = np.full(np.shape(label_im), i + 1)

        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int)

        # replace 1 with 255 for visualization as rgb image

        separate_mask[separate_mask == 1] = 255
        # append separate mask to the mask_array
        mask_array.append(separate_mask)

    return mask_array


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


def get_tile_mean(img_dir):
    """ function to calculate mean of images for preprocessing with yml file """

    # Access all PNG files in directory
    imlist = [image_file for image_file in glob.iglob('%s/**/*.png' % img_dir, recursive=True)]

    # Get the sum and count of pixel values across all images
    total_sum = 0
    total_count = 0
    chan1 = []
    chan2 = []
    chan3 = []
    for im in imlist:
        cur_img = mpimg.imread(im)
        cur_sum = np.sum(cur_img, axis=tuple(range(cur_img.ndim - 1)))
        cur_count = np.shape(cur_img)[0] * np.shape(cur_img)[1]
        total_sum += cur_sum
        total_count += cur_count

        # appending all values for std dev (might need to rework this for large sample)
        chan1.append(cur_img[0][0][0])
        chan2.append(cur_img[0][0][1])
        chan3.append(cur_img[0][0][2])

    # Store Values into Numpy array then split numpy array
    avg_pix = total_sum / total_count

    # Set values to RGB(Ignore channel 4)
    y = np.split(avg_pix, 4)
    R = float(y[0] * 255)
    G = float(y[1] * 255)
    B = float(y[2] * 255)

    # Return the mean for images channels
    return (R, G, B)




if __name__ == '__main__':
    # Using the project path sets us to the root of the repository.
    use_project_path()

    # Temporary configuration variables will be contained here.  They will hopefully be updated to runtime
    # arguments once everything is working
    TRAINING_COLLECTION = 'sample2'
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

    # Get Mean Pixel Values of Images being loaded in
    image_path = 'temp_data/%s/tile-256' % TRAINING_COLLECTION
    R, G, B = get_tile_mean(image_path)


    # Create the mrcnn configuration object.  We are adjusting everything to understand batch size and using
    # the training and validation lengths as read back in from the pandas dataframe
    config = BuildingConfig()
    config.STEPS_PER_EPOCH = int(training_image_count / BATCH_SIZE)
    config.VALIDATION_STEPS = int(validation_image_count / BATCH_SIZE)
    config.MEAN_PIXEL = np.array([R, G, B]) # Use Mean Pixel of dataset being loaded in
    config.display()

    # Load the training image dataset.
    dataset_train = BuildingDataset(training_images)
    dataset_train.load_building()
    dataset_train.prepare()

    # Load validation dataset, done so by setting validation to true
    dataset_val = BuildingDataset(training_images)
    dataset_val.load_building(validation=True)
    dataset_val.prepare()

    # Create model in inference
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    # For now, we are just going to find the last weights file.
    model_path = 'temp_data/logs/mask_rcnn_building_0080.h5'
    model.load_weights(model_path, by_name=True)

    #
    # Note for Sebastian: I have no idea how well the rest of this code works...I haven't gotten far enough to
    # test it out just yet.
    #

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)

    print("This is the image_ID", str(image_id)) #Find out which image is showing
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
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
