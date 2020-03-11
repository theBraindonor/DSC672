#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
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
from PIL import Image

from utility import use_project_path
from utility import MaskRCNNBuildingConfig
from utility import MaskRCNNBuildingDataset
from utility import create_mrcnn_training_images_split


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataframe', default=None,
                        help='Pandas DataFrame containing the training and testing images.')
    parser.add_argument('-mp', '--model-path', default='temp_data/mask_rcnn_building_0040_v3.h5',
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
    parser.add_argument('-t', '--type', default='score',
                        help='The kind of test to run, demo/score/predict')
    parser.add_argument('-th', '--threshold', default=0.9,
                        help='The threshold to use in predictions')
    parser.add_argument('-o', '--output', default='temp_data/maskrcnn_output',
                        help='Output folder for binary segmentation images')
    arguments = vars(parser.parse_args())

    DATA_FRAME = arguments['dataframe']
    MODEL_PATH = arguments['model_path']
    IMAGE_PATH = arguments['image_path']
    COLLECTION = arguments['collection']
    VALIDATION_SIZE = float(arguments['validation'])
    COUNT = int(arguments['number'])
    RANDOM_SEED = int(arguments['random_seed'])
    TYPE = arguments['type']
    THRESHOLD = float(arguments['threshold'])
    OUTPUT = arguments['output']

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
    print('       Test Type: %s' % TYPE)
    print('       Threshold: %s' % THRESHOLD)
    print('     Output Path: %s' % OUTPUT)
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
        training_images = create_mrcnn_training_images_split(IMAGE_PATH, COLLECTION, VALIDATION_SIZE)
    training_image_count = len(training_images[(training_images['validation'] == 0.0)])
    validation_image_count = len(training_images[(training_images['validation'] == 1.0)])

    # Create the mrcnn configuration object.  We are adjusting everything to understand batch size and using
    # the training and validation lengths as read back in from the pandas dataframe
    config = MaskRCNNBuildingConfig()
    config.DETECTION_MIN_CONFIDENCE = THRESHOLD
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

    # Demonstrate the model by showing the original image and the generated bounding boxes and mask.
    if TYPE == 'demo':
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

    if TYPE == 'score' or TYPE == 'predict':
        Path(OUTPUT).mkdir(parents=True, exist_ok=True)
        # Get Image outputs from MASK RCNN
        dataset_val_images = len(dataset_val.image_ids)

        #Create a Pandas DF to hold values and output CSV file after output is done
        if TYPE == 'score':
            Jvalues = [] # Hold Jaccard Score Values
            fName = [] # Hold File Names
            mask_intersection_pixels = []
            actual_mask_pixels = []
            pred_mask_pixels = []

        for i in range(dataset_val_images):
            # Get Image ID
            image_id = i

            # clean up image name for saving
            temp = dataset_val.get_image_filename(image_id)
            print('Processing file: %s' % temp)
            print('File %s of %s' % (i + 1, dataset_val_images))
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
                                 filter_classs_names=['building'], scores_thresh=0.7, mode=3,
                                 save_dir=OUTPUT)

            # If turned on, get score of pred and ground truth
            if TYPE == 'score':
                actual_mask = skimage.io.imread(os.path.join(mask_name))
                pred_mask = skimage.io.imread(os.path.join(OUTPUT,str(image_name + ".png")))
                Jaccard = (jaccard_score(actual_mask, pred_mask, average='micro'))
                fName.append(temp)

                # Grab the needed pixel counts for a global jaccard score.
                actual_mask_pixels.append(np.sum(actual_mask > 0))
                pred_mask_pixels.append(np.sum(pred_mask > 0))
                mask_intersection_pixels.append(np.sum(np.logical_and(actual_mask > 0, pred_mask > 0)))

                # Zero Division Check
                if Jaccard == 0:
                    print(gt_mask.shape)
                    print(r['masks'].shape)
                    # Check to see if mask has any buildings and predicted mask has buildings
                    # if niether have buildings consider this a 1
                    #r['masks] shape has to be (0,28,28) due to mask shape output
                    if gt_mask.shape == (256,256,0) and r['masks'].shape == (0,28,28):
                        Jaccard = 1

                print('Score: %s' % Jaccard)
                Jvalues.append(Jaccard)


        # Output CSV file
        if TYPE == 'score':
            JaccardDF = pd.DataFrame()
            JaccardDF['File_Name'] = pd.Series(fName)
            JaccardDF['Scores'] = pd.Series(Jvalues)
            JaccardDF['IntersectionPixels'] = pd.Series(mask_intersection_pixels)
            JaccardDF['ActualMaskPixels'] = pd.Series(actual_mask_pixels)
            JaccardDF['PredMaskPixels'] = pd.Series(pred_mask_pixels)
            output_file = os.path.join("temp_data", 'MaskRCNN_Scores.csv')
            JaccardDF.to_csv(output_file, index=False, header=True)

            # Global Jaccard score is the sum of intersected pixels over the sum of unioned pixels from each tile.
            global_jaccard = np.sum(mask_intersection_pixels) / \
                             (np.sum(actual_mask_pixels) + np.sum(pred_mask_pixels) - np.sum(mask_intersection_pixels))

            # Output mean score
            print('')
            print('  Mean Jaccard Score: %s' % np.mean(Jvalues))
            print('Global Jaccard Score: %s' % global_jaccard)
            print('')

    # Turn this on if you want to run submission for Test set:
    submit_Pred = False
    if submit_Pred == True:
        Path('raw_source_data/maskrcnn_output').mkdir(parents=True, exist_ok=True)

        ims = [f for f in os.listdir('raw_source_data/%s/tile-256' % "test")]
        x = 0
        for im in ims:
            x += 1
            print(im)
            # print(x)
            if im == ".DS_Store":  # MAC Error Ignore this
                print("Ignore")
            else:
                t_image = skimage.io.imread(os.path.join('raw_source_data/%s/tile-256/' % 'test', im))

                im = im[:-4]
                # Remove Alpha Channel
                t_image = t_image[:, :, :3]
                results = model.detect([t_image], verbose=1)
                r = results[0]

                visualize.save_image(t_image, im, r['rois'], r['masks'],
                                     r['class_ids'], r['scores'], dataset_val.class_names,
                                     filter_classs_names=['building'], scores_thresh=0.7, mode=3,
                                     save_dir='raw_source_data/maskrcnn_output/')

        print("Converting to .TIFF")
        # Re size images and save as TIFF for submission
        ims = [f for f in os.listdir('raw_source_data/maskrcnn_output')]
        Path('raw_source_data/submission').mkdir(parents=True, exist_ok=True)
        for im in ims:
            if im == ".DS_Store":  # Mac Error
                print("Ignore")
            else:
                p_image = skimage.io.imread(os.path.join('raw_source_data/maskrcnn_output/', im))
                p_image = Image.fromarray(p_image)
                p_image = p_image.resize((1024, 1024))
                im = im[:-4]  # remove extension
                submission_file = os.path.join('raw_source_data/submission/', im + ".TIFF")
                p_image.save(submission_file)