

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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)

from os import listdir
from os.path import isfile, join
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Results Directory
# Save Submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "Results/Buildings/")
# Currently a temp set up, just identify valdiation set, contains 171 images from the new formed zoom 20 sample training set
VAL_IMAGE_IDS = ['nia_825a50_396_530115_484370_20.png','nia_825a50_397_530116_484370_20.png','nia_825a50_398_530117_484370_20.png','nia_825a50_399_530118_484370_20.png',
                 'nia_825a50_400_530119_484370_20.png','nia_825a50_401_530120_484370_20.png','nia_825a50_402_530121_484370_20.png','nia_825a50_403_530122_484370_20.png',
                 'nia_825a50_404_530123_484370_20.png','nia_825a50_405_530124_484370_20.png','nia_825a50_406_530125_484370_20.png','nia_825a50_407_530126_484370_20.png',
                 'nia_825a50_408_530127_484370_20.png','nia_825a50_409_530128_484370_20.png','nia_825a50_410_530129_484370_20.png','nia_825a50_411_530130_484370_20.png',
                 'nia_825a50_412_530131_484370_20.png','nia_825a50_413_530132_484370_20.png','nia_825a50_414_530133_484370_20.png','nia_825a50_415_530115_484371_20.png',
                 'nia_825a50_416_530116_484371_20.png','nia_825a50_417_530117_484371_20.png','nia_825a50_418_530118_484371_20.png','nia_825a50_419_530119_484371_20.png',
                 'nia_825a50_420_530120_484371_20.png','nia_825a50_421_530121_484371_20.png','nia_825a50_422_530122_484371_20.png','nia_825a50_423_530123_484371_20.png',
                 'nia_825a50_424_530124_484371_20.png','nia_825a50_425_530125_484371_20.png','nia_825a50_426_530126_484371_20.png','nia_825a50_427_530127_484371_20.png',
                 'nia_825a50_428_530128_484371_20.png','nia_825a50_429_530129_484371_20.png','nia_825a50_430_530130_484371_20.png','nia_825a50_431_530131_484371_20.png',
                 'nia_825a50_432_530115_484372_20.png','nia_825a50_433_530116_484372_20.png','nia_825a50_434_530117_484372_20.png','nia_825a50_435_530118_484372_20.png',
                 'nia_825a50_436_530119_484372_20.png','nia_825a50_437_530120_484372_20.png','nia_825a50_438_530121_484372_20.png','nia_825a50_439_530122_484372_20.png',
                 'nia_825a50_440_530123_484372_20.png','nia_825a50_441_530124_484372_20.png','nia_825a50_442_530125_484372_20.png','nia_825a50_443_530126_484372_20.png',
                 'nia_825a50_444_530127_484372_20.png','nia_825a50_445_530128_484372_20.png','nia_825a50_446_530129_484372_20.png','nia_825a50_447_530116_484373_20.png',
                 'nia_825a50_448_530117_484373_20.png','nia_825a50_449_530118_484373_20.png','nia_825a50_450_530119_484373_20.png','nia_825a50_451_530120_484373_20.png',
                 'nia_825a50_452_530121_484373_20.png','nia_825a50_453_530122_484373_20.png','nia_825a50_454_530123_484373_20.png','nia_825a50_455_530124_484373_20.png',
                 'nia_825a50_456_530125_484373_20.png','nia_825a50_457_530126_484373_20.png','nia_825a50_458_530127_484373_20.png','nia_825a50_459_530116_484374_20.png',
                 'nia_825a50_460_530117_484374_20.png','nia_825a50_461_530118_484374_20.png','nia_825a50_462_530119_484374_20.png','nia_825a50_463_530120_484374_20.png',
                 'nia_825a50_464_530121_484374_20.png','nia_825a50_465_530122_484374_20.png','nia_825a50_466_530123_484374_20.png','nia_825a50_467_530124_484374_20.png',
                 'nia_825a50_468_530125_484374_20.png','nia_825a50_469_530126_484374_20.png','nia_825a50_470_530116_484375_20.png','nia_825a50_471_530117_484375_20.png',
                 'nia_825a50_472_530118_484375_20.png','nia_825a50_473_530119_484375_20.png','nia_825a50_474_530120_484375_20.png','nia_825a50_475_530121_484375_20.png',
                 'nia_825a50_476_530122_484375_20.png','nia_825a50_477_530123_484375_20.png','nia_825a50_478_530124_484375_20.png','nia_825a50_479_530125_484375_20.png',
                 'nia_825a50_480_530126_484375_20.png','nia_825a50_481_530117_484376_20.png','nia_825a50_482_530118_484376_20.png','nia_825a50_483_530119_484376_20.png',
                 'nia_825a50_484_530120_484376_20.png','nia_825a50_485_530121_484376_20.png','nia_825a50_486_530122_484376_20.png','nia_825a50_487_530123_484376_20.png',
                 'nia_825a50_488_530124_484376_20.png','nia_825a50_489_530125_484376_20.png','nia_825a50_490_530126_484376_20.png','nia_825a50_491_530127_484376_20.png',
                 'nia_825a50_492_530118_484377_20.png','nia_825a50_493_530119_484377_20.png','nia_825a50_494_530120_484377_20.png','nia_825a50_495_530121_484377_20.png',
                 'nia_825a50_496_530122_484377_20.png','nia_825a50_497_530123_484377_20.png','nia_825a50_498_530124_484377_20.png','nia_825a50_499_530125_484377_20.png',
                 'nia_825a50_500_530126_484377_20.png','nia_825a50_501_530127_484377_20.png','nia_825a50_502_530128_484377_20.png','nia_825a50_503_530118_484378_20.png',
                 'nia_825a50_504_530119_484378_20.png','nia_825a50_505_530120_484378_20.png','nia_825a50_506_530121_484378_20.png','nia_825a50_507_530122_484378_20.png',
                 'nia_825a50_508_530123_484378_20.png','nia_825a50_509_530124_484378_20.png','nia_825a50_510_530125_484378_20.png','nia_825a50_511_530126_484378_20.png',
                 'nia_825a50_512_530127_484378_20.png','nia_825a50_513_530128_484378_20.png','nia_825a50_514_530129_484378_20.png','nia_825a50_515_530119_484379_20.png',
                 'nia_825a50_516_530120_484379_20.png','nia_825a50_517_530121_484379_20.png','nia_825a50_518_530122_484379_20.png','nia_825a50_519_530123_484379_20.png',
                 'nia_825a50_520_530124_484379_20.png','nia_825a50_521_530125_484379_20.png','nia_825a50_522_530126_484379_20.png','nia_825a50_523_530127_484379_20.png',
                 'nia_825a50_524_530128_484379_20.png','nia_825a50_525_530129_484379_20.png','nia_825a50_526_530120_484380_20.png','nia_825a50_527_530121_484380_20.png',
                 'nia_825a50_528_530122_484380_20.png','nia_825a50_529_530123_484380_20.png','nia_825a50_530_530124_484380_20.png','nia_825a50_531_530125_484380_20.png',
                 'nia_825a50_532_530126_484380_20.png','nia_825a50_533_530127_484380_20.png','nia_825a50_534_530128_484380_20.png','nia_825a50_535_530129_484380_20.png',
                 'nia_825a50_536_530121_484381_20.png','nia_825a50_537_530122_484381_20.png','nia_825a50_538_530123_484381_20.png','nia_825a50_539_530124_484381_20.png',
                 'nia_825a50_540_530125_484381_20.png','nia_825a50_541_530126_484381_20.png','nia_825a50_542_530127_484381_20.png','nia_825a50_543_530128_484381_20.png',
                 'nia_825a50_544_530129_484381_20.png','nia_825a50_545_530122_484382_20.png','nia_825a50_546_530123_484382_20.png','nia_825a50_547_530124_484382_20.png',
                 'nia_825a50_548_530125_484382_20.png','nia_825a50_549_530126_484382_20.png','nia_825a50_550_530127_484382_20.png','nia_825a50_551_530128_484382_20.png',
                 'nia_825a50_552_530123_484383_20.png','nia_825a50_553_530124_484383_20.png','nia_825a50_554_530125_484383_20.png','nia_825a50_555_530126_484383_20.png',
                 'nia_825a50_556_530127_484383_20.png','nia_825a50_557_530128_484383_20.png','nia_825a50_558_530124_484384_20.png','nia_825a50_559_530125_484384_20.png',
                 'nia_825a50_560_530126_484384_20.png','nia_825a50_561_530127_484384_20.png','nia_825a50_562_530125_484385_20.png','nia_825a50_563_530126_484385_20.png',
                 'nia_825a50_564_530127_484385_20.png','nia_825a50_565_530126_484386_20.png','nia_825a50_566_530127_484386_20.png',]
#################################################
# Configurations
#################################################

class BuildingConfig(Config):
    """Configuration for training on the sample Building  dataset.
    Derives from the base Config class and overrides values specific
    to the building shapes dataset. See the CONFIG FILE for more attributes
    """

    #Name config
    NAME = "Building"

    #Train on 1 GPU and 8 images per PGU. We can put multiple images on each
    #GPU because the images are small. Batch size is 8 (GPUs *images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    #Number of classes (including background)
    NUM_CLASSES = 1 + 1 #1 background + 1 Building class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = BuildingConfig()
config.display()

#################################################
# Dataset
#################################################

## TOOK TEMPLATE SAMPLE FROM Nucleus.py was dealing with PNG images, seemed related to our file setup
class BuildingDataset(utils.Dataset):

    def load_building(self, dataset_dir, subset):
        """Load a subset of the building dataset.
               dataset_dir: Root directory of the dataset.
               subset: Subset to load: train or val
               """
        # This will need to be changed based on the images loaded
        #dataset_dir = "/Users/Sebastian/Documents/GitHub/DSC672/"
        # Add classes. We have only one class to add.
        self.add_class("building", 1, "building")



        # Which Dataset are you loading in?
         # Val:Use validation test est
         # train : use data from temp_data/sample/tile -256
         # TO DO: ADD other training/testing sources
        assert subset in ["sample/", "val"]
        subset_dir = "sample/tile-256/"
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image Ids from directory names
            # print(dataset_dir) ## Remove latter DEBUG
            image_ids = next(os.walk(dataset_dir))[2]
            # image_ids = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
            # print(image_ids) ## Remove Later DEBUG
            if subset == "sample/":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # dir_name = join(dataset_dir + "temp_data/sample/tile-256") # for loading in regualr tiles
        # annotations_b = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

        # dir_name = join(dataset_dir + "temp_data/sample/mask-256") # for loading in mask tiles
        # annotations_m = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

        #Add images
        for image_id in image_ids:
            self.add_image(
                "building",
                image_id=image_id,
                path = os.path.join(dataset_dir, image_id.format(image_id)))


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "mask-256")
        print("this is ", mask_dir) # REMOVE LATER FOR DEBUG
        # Read mask files from .png image
        mask = []
        # image_ids = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
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





# Training Dataset
dataset_train = BuildingDataset()

dataset_dir = "/Users/Sebastian/Documents/GitHub/DSC672/temp_data/"
subset = "sample/"
dataset_train.load_building(dataset_dir, subset)
dataset_train.prepare()

print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))



# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit= 1) #Coded provided to visualize outputs
    # display sample results
    print(image_id)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image Tile')
    print(image.shape)
    # Fix Mask Shape
    #if mask.shape[-1] == 160: #Seems that Masks are stacking on each other, shape issue
       # print(mask.shape)
       # mask = mask[..., :3]
       # print(mask.shape)
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Actual Image Mask')
    plt.show()


###########################################
# TRAINING



###########################################
## TODO: CURRENTLY MASK IMAGES ARE STACKING ONE ANOTHER, SHAPE SHOULD BE 256,256,1 BUT ITS COMING IN AS 256,256,160
## TODO: SETUP INFRENCE MODEL BUILD FOR RUNING ON TESTING SET
## TODO: NUCLEUS.PY HAS A TUTORIAL ON HOW TO BUILD MODEL FOR INFRENCE CURRENTLY REVIEWING
## TODO: CLEAN UP PATHS WITHIN FILE TO BE PROJECT FREIENDLY












