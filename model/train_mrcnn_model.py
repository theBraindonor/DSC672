

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
    STEPS_PER_EPOCH = 4
    VALIDATION_STEPS = 1
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9]) # TODO: Update with better means


class BuildingDataset(utils.Dataset):

    def load_building(self, dataset_dir, subset):
        print("Loading buildings...")
        """Load a subset of the building dataset.
               dataset_dir: Root directory of the dataset.
               subset: Subset to load: train or val
               """
        # This will need to be changed based on the images loaded
        # dataset_dir = "/Users/Sebastian/Documents/GitHub/DSC672/"
        # Add classes. We have only one class to add.
        self.add_class("building", 1, "building")

        # Which Dataset are you loading in?
        # Val:Use validation test est
        # train : use data from temp_data/sample/tile -256
        # TO DO: ADD other training/testing sources
        assert subset in ["sample_lg/", "val"]
        subset_dir = "sample_lg/tile-256/"
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image Ids from directory names
            # print(dataset_dir) ## Remove latter DEBUG
            image_ids = next(os.walk(dataset_dir))[2]
            # image_ids = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
            # print(image_ids) ## Remove Later DEBUG
            if subset == "sample_lg/":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # dir_name = join(dataset_dir + "temp_data/sample/tile-256") # for loading in regualr tiles
        # annotations_b = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

        # dir_name = join(dataset_dir + "temp_data/sample/mask-256") # for loading in mask tiles
        # annotations_m = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

        # Add images
        for image_id in image_ids:
            print("Adding Image!")
            print("building")
            print(image_id)
            print(os.path.join(dataset_dir, image_id.format(image_id)))

            self.add_image(
                "building",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id.format(image_id)))

        print("DONE!")

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # print("This is info ",str(info))
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask-256")
        # print("this is ", mask_dir) # REMOVE LATER FOR DEBUG
        # Read mask files from .png image
        mask = []
        # image_ids = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
        for f in next(os.walk(mask_dir))[2]:
            if f == info['id']:  # match image id to mask id
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)

        # print("masks has {} items".format(len(mask)))
        mask = np.stack(mask, axis=-1)
        # print("mask's shape is {}".format(mask.shape))
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
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


if __name__ == '__main__':
    # Using the project path sets us to the root of the repository.
    use_project_path()

    # This model will make use of the coco weights provided by the mask_rcnn project.
    COCO_MODEL_PATH = 'temp_data/mask_rcnn_coco.h5'
    if not os.path.exists(COCO_MODEL_PATH):
        print("Downloading Coco Weights...")
        utils.download_trained_weights(COCO_MODEL_PATH)

    exit(0)

    # Results Directory
    # Save Submission files here
    RESULTS_DIR = os.path.join(ROOT_DIR, "Results/Buildings/")
    # Currently a temp set up, just identify valdiation set, contains 171 images from the new formed zoom 20 sample training set
    VAL_IMAGE_IDS = ['nia_825a50_412_530131_484370_20.png','nia_825a50_413_530132_484370_20.png','nia_825a50_414_530133_484370_20.png','nia_825a50_415_530115_484371_20.png']
    #################################################
    # Configurations
    #################################################







    config = BuildingConfig()
    config.display()

    #################################################
    # Dataset
    #################################################






    ############################################
    # DATA SET LOAD IN

    # Training Dataset
    dataset_train = BuildingDataset()

    dataset_dir = "D:/Repos/Depaul/DSC672/temp_data/"
    subset = "sample_lg/"
    dataset_train.load_building(dataset_dir, subset)
    dataset_train.prepare()

    # Print out dataset information regarding the training set
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))



    # Load validation dataset
    dataset_val = BuildingDataset()
    dataset_val.load_building(dataset_dir, "val")
    dataset_val.prepare()

    # Print out Validation dataset info
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))



    # Load and display random samples
    #image_ids = np.random.choice(dataset_train.image_ids, 4)
    #for image_id in image_ids:
      #  image = dataset_train.load_image(image_id)
       # mask, class_ids = dataset_train.load_mask(image_id)
       # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit= 1) #Coded provided to visualize outputs

    ############################################


    ###########################################
    # TRAINING

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)



    #### Turn this on if you want to train the model
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='heads')


    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='4+')



    #### Turn this on if you want to train the model
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=10,
                layers="all")


    ###########################################
    # INFERENCE MODE
    inference_config = config

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # /Users/Sebastian/Documents/GitHub/logs/building20200209T1400
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    log_dir = "D:/Repos/DePaul/DSC672/temp_data/logs/building20200213T2133/"
    model_path = os.path.join(log_dir, "mask_rcnn_building_0005.h5")
    # model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # Test on a random image
    # Image_ID = 59
    # Image_ID = 117 street sample
    # Image Id = 42 Building Sample
    # Image ID = 44
    image_id = random.choice(dataset_val.image_ids)
    image_id = 44

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
