#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import re

import pandas as pd
import os

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


from sklearn.metrics import jaccard_score # this needs to be installed for the environment
import skimage

from utility import use_project_path
from utility import MaskRCNNBuildingConfig
from utility import MaskRCNNBuildingDataset
from utility import create_mrcnn_training_images_split


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataframe', default=None,
                        help='Pandas DataFrame containing the training and testing images.')
    parser.add_argument('-mp', '--model-path', default='temp_data/logs/mask_rcnn_building_0040.h5',
                        help='Path to the Mask R-CNN Model')
    parser.add_argument('-c', '--collection', default='sample_lg',
                        help='The image collection to load')
    parser.add_argument('-ip', '--image-path', default='temp_data',
                        help='The folder containing the image collections')
    parser.add_argument('-v', '--validation', default=0.2,
                        help='The size of the validation set')
    parser.add_argument('-n', '--number', default=1,
                        help='The number of random images to display')
    parser.add_argument('-rs', '--random-seed', default=112,
                        help='Random seed for sample images.')
    arguments = vars(parser.parse_args())

    DATA_FRAME = arguments['dataframe']
    MODEL_PATH = arguments['model_path']
    IMAGE_PATH = arguments['image_path']
    COLLECTION = arguments['collection']
    VALIDATION_SIZE = float(arguments['validation'])
    COUNT = int(arguments['number'])
    RANDOM_SEED = int(arguments['random_seed'])

    print('')
    print('Quick Testing RUN of Mask R-CNN Model...')
    print('')
    print('Parameters:')
    print('      Data Frame: %s' % DATA_FRAME)
    print('      Model Path: %s' % MODEL_PATH)
    print('      Collection: %s' % COLLECTION)
    print('      Image Path: %s' % IMAGE_PATH)
    print(' Validation Size: %s' % VALIDATION_SIZE)
    print('           Count: %s' % COUNT)
    print('     Random Seed: %s' % RANDOM_SEED)
    print('')

    MODEL_DIR = 'temp_data/logs'

    # Using the project path sets us to the root of the repository.
    use_project_path()

    # Create a pandas array of training images.  Please note that this will perform a default 80/20 training
    # and testing split.  If we had a DATA_FRAME specified in the arguments call, then we will instead load
    # that data frame.
    if DATA_FRAME is not None:
        training_images = pd.read_csv(DATA_FRAME)
        pass
    else:
        training_images = create_mrcnn_training_images_split(IMAGE_PATH, TRAINING_COLLECTION, VALIDATION_SIZE)
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
    model.load_weights(MODEL_PATH, by_name=True)

    # Turn this on true to test and see an image
    show_test = False

    if show_test == True:
        random.seed(RANDOM_SEED)
        for i in range(COUNT):
            # Test on a random image
            image_id = random.choice(dataset_val.image_ids)

            print("This is the image_ID", str(image_id)) #Find out which image is showing
            print("Filename: %s" % dataset_val.get_image_filename(image_id))
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


    # Turn this to True if you want to output mask images
    output = True

    # Turn this on to get a Jaccard Score file output
    Jscore = True

    if output == True:
        # Get Image outputs from MASK RCNN
        dataset_val_images = len(dataset_val.image_ids)

        #If score is turned on, Create a Pandas DF to hold values and output CSV file after output is done
        if Jscore == True:
            Jvalues = [] # Hold Jaccard Score Values
            fName = [] # Hold File Names

        for i in range(dataset_val_images):
            # Get Image ID
            image_id = i

            # clean up image name for saving
            temp = dataset_val.get_image_filename(image_id)
            mask_name = re.sub(r'.tile-256','/mask-256',temp)
            #image_name = dataset_val.get_image_filename(image_id)
            image_name = temp[:-4]
            image_name = re.sub(r'.*256/', '', image_name)

            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, config,
                                       image_id, use_mini_mask=False)

            log("original_image", original_image)
            log("image_meta", image_meta)
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)

            # DETECT MASK FOR  IMAGE
            results = model.detect([original_image], verbose=1)

            r = results[0]

            # Function used to save mask predictions
            # Mode 3 sets it to a black background with only mask segmentation
            # creates an output directory for images to be saved in
            visualize.save_image(original_image, image_name, r['rois'], r['masks'],
                                 r['class_ids'], r['scores'], dataset_val.class_names,
                                 filter_classs_names=['building'], scores_thresh=0.7, mode=3)

            # If turned on, get score of pred and ground truth
            if Jscore == True:
                actual_mask = skimage.io.imread(os.path.join(mask_name))
                pred_mask = skimage.io.imread(os.path.join("output/",str(image_name + ".png")))
                Jaccard = (jaccard_score(actual_mask, pred_mask, average='micro'))
                fName.append(temp)

                # Zero Division Check
                if Jaccard == 0:
                    print(gt_mask.shape)
                    print(r['masks'].shape)
                    # Check to see if mask has any buildings and predicted mask has buildings
                    # if niether have buildings consider this a 1
                    #r['masks] shape has to be (0,28,28) due to mask shape output
                    if gt_mask.shape == (256,256,0) and r['masks'].shape == (0,28,28):
                        Jaccard = 1

                Jvalues.append(Jaccard)


        # Output CSV file
        if Jscore == True:
            JaccardDF = pd.DataFrame()
            JaccardDF['File_Name'] = pd.Series(fName)
            JaccardDF['Scores'] = pd.Series(Jvalues)
            output_file = os.path.join("temp_data", 'MaskRCNN_Scores.csv')
            JaccardDF.to_csv(output_file, index=False, header=True)


