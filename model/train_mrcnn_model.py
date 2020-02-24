#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os

from mrcnn import utils
import mrcnn.model as modellib

from utility import use_project_path
from utility import MaskRCNNBuildingConfig
from utility import MaskRCNNBuildingDataset
from utility import create_mrcnn_training_images_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', default='temp_data',
                        help='The folder containing the image collections.')
    parser.add_argument('-c', '--collection', default='sample',
                        help='The image collection to load.')
    parser.add_argument('-v', '--validation', default=0.2,
                        help='The size of the validation set.')
    parser.add_argument('-ipg', '--images-per-gpu', default=1,
                        help='The number of images to batch to the GPU.')
    parser.add_argument('-spe', '--steps-per-epoch', default=50,
                        help='The number of training steps to perform.')
    parser.add_argument('-vs', '--validation-steps', default=5,
                        help='The number of validation steps to perform.')
    parser.add_argument('-i', '--init-with', default='coco',
                        help='How to initialize the model for training.')
    parser.add_argument('-eh', '--epochs-head', default=1,
                        help='Number of training epochs for the head layers')
    parser.add_argument('-er', '--epochs-resnet', default=1,
                        help='Number of training epochs for the resnet layers.')
    parser.add_argument('-ea', '--epochs-all', default=1,
                        help='Number of training epochs for all layers.')
    parser.add_argument('-md', '--model-dir', default='temp_data/logs',
                        help='Directory to store model logs and checkpoints.')
    parser.add_argument('-cmp', '--coco-model-path', default='temp_data/mask_rcnn_coco.h5',
                        help='Path to the COCO model weights.')
    arguments = vars(parser.parse_args())

    # Using the project path sets us to the root of the repository.

    # Temporary configuration variables will be contained here.  They will hopefully be updated to runtime
    # arguments once everything is working
    IMAGE_PATH = arguments['image_path']
    TRAINING_COLLECTION = arguments['collection']
    VALIDATION_SIZE = float(arguments['validation'])
    IMAGES_PER_GPU = int(arguments['images_per_gpu'])
    STEPS_PER_EPOCH = int(arguments['steps_per_epoch'])
    VALIDATION_STEPS = int(arguments['validation_steps'])
    INITIALIZE_WITH = arguments['init_with']
    EPOCHS_HEAD = int(arguments['epochs_head'])
    EPOCHS_RESNET = int(arguments['epochs_resnet'])
    EPOCHS_ALL = int(arguments['epochs_all'])
    MODEL_DIR = arguments['model_dir']
    COCO_MODEL_PATH = arguments['coco_model_path']

    print('')
    print('Starting a Training Run of the Mask R-CNN Model...')
    print('')
    print('Parameters:')
    print('          Image Path: %s' % IMAGE_PATH)
    print('          Collection: %s' % TRAINING_COLLECTION)
    print('     Validation Size: %s' % VALIDATION_SIZE)
    print('      Images Per GPU: %s' % IMAGES_PER_GPU)
    print('     Steps Per Epoch: %s' % STEPS_PER_EPOCH)
    print('    Validation Steps: %s' % VALIDATION_STEPS)
    print('     Initialize With: %s' % INITIALIZE_WITH)
    print('         Epochs Head: %s' % EPOCHS_HEAD)
    print('       Epochs ResNet: %s' % EPOCHS_RESNET)
    print('          Epochs All: %s' % EPOCHS_ALL)
    print('     Model Directory: %s' % MODEL_DIR)
    print('     COCO Model Path: %s' % COCO_MODEL_PATH)
    print('')

    use_project_path()

    # Ensure we have a model directory to write to
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # This model will make use of the coco weights provided by the mask_rcnn project.
    if not os.path.exists(COCO_MODEL_PATH) and INITIALIZE_WITH == 'coco':
        print("Downloading Coco Weights...")
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Create a pandas array of training images.  Please note that this will perform a default 80/20 training
    # and testing split.
    training_images = create_mrcnn_training_images_split(IMAGE_PATH, TRAINING_COLLECTION, VALIDATION_SIZE)
    training_image_count = len(training_images[(training_images['validation'] == 0.0)])
    validation_image_count = len(training_images[(training_images['validation'] == 1.0)])

    # Create the mrcnn configuration object and update with command line variables.
    config = MaskRCNNBuildingConfig()
    config.IMAGES_PER_GPU = IMAGES_PER_GPU
    config.STEPS_PER_EPOCH = STEPS_PER_EPOCH
    config.VALIDATION_STEPS = VALIDATION_STEPS
    config.display()

    # Load the training image dataset.
    dataset_train = MaskRCNNBuildingDataset(training_images)
    dataset_train.load_building()
    dataset_train.prepare()

    # Load validation dataset, done so by setting validation to true
    dataset_val = MaskRCNNBuildingDataset(training_images)
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
        model.load_weights(model.find_last()[0], by_name=True)

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

    # Training - Stage 3
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Training All Layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=(EPOCHS_HEAD + EPOCHS_RESNET + EPOCHS_ALL),
                layers="all")
