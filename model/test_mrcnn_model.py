#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from utility import use_project_path
from utility import MaskRCNNBuildingConfig
from utility import MaskRCNNBuildingDataset
from utility import create_mrcnn_training_images_split


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model-path', default='temp_data/mask_rcnn_building_0080.h5',
                        help='Path to the Mask R-CNN Model')
    parser.add_argument('-c', '--collection', default='sample_lg',
                        help='The image collection to load')
    parser.add_argument('-ip', '--image-path', default='temp_data',
                        help='The folder containing the image collections')
    parser.add_argument('-v', '--validation', default=0.2,
                        help='The size of the validation set')
    parser.add_argument('-n', '--number', default=1,
                        help='The number of random images to display')
    arguments = vars(parser.parse_args())

    MODEL_PATH = arguments['model_path']
    IMAGE_PATH = arguments['image_path']
    COLLECTION = arguments['collection']
    VALIDATION_SIZE = float(arguments['validation'])
    COUNT = int(arguments['number'])

    print('')
    print('Quick Testing RUN of Mask R-CNN Model...')
    print('')
    print('Parameters:')
    print('      Model Path: %s' % MODEL_PATH)
    print('      Collection: %s' % COLLECTION)
    print('      Image Path: %s' % IMAGE_PATH)
    print(' Validation Size: %s' % VALIDATION_SIZE)
    print('           Count: %s' % COUNT)
    print('')

    MODEL_DIR = 'temp_data/logs'

    # Using the project path sets us to the root of the repository.
    use_project_path()

    # Create a pandas array of training images.  Please note that this will perform a default 80/20 training
    # and testing split.
    training_images = create_mrcnn_training_images_split(IMAGE_PATH, COLLECTION, VALIDATION_SIZE)
    training_image_count = len(training_images[(training_images['validation'] == 0.0)])
    validation_image_count = len(training_images[(training_images['validation'] == 1.0)])

    # Create the mrcnn configuration object.  We are adjusting everything to understand batch size and using
    # the training and validation lengths as read back in from the pandas dataframe
    config = MaskRCNNBuildingConfig()
    config.display()

    # Load the training image dataset.
    dataset_train = MaskRCNNBuildingDataset(training_images)
    dataset_train.load_building()
    dataset_train.prepare()

    # Load validation dataset, done so by setting validation to true
    dataset_val = MaskRCNNBuildingDataset(training_images)
    dataset_val.load_building(validation=True)
    dataset_val.prepare()

    # Create model in inference
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    # For now, we are just going to find the last weights file.
    model_path = 'temp_data/mask_rcnn_building_0080.h5'
    model.load_weights(model_path, by_name=True)

    #
    # Note for Sebastian: I have no idea how well the rest of this code works...I haven't gotten far enough to
    # test it out just yet.
    #

    for i in range(COUNT):
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
