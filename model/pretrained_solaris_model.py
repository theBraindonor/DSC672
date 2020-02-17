
import solaris as sol
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import glob
import os

from utility import use_project_path

import yaml # this needs to be installed for the environment
from sklearn.metrics import jaccard_score # this needs to be installed for the environment


def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
        [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
    ]))

    training_df = pd.DataFrame(file_array, columns=['image', 'label'])
    training_df.to_csv('temp_data/solaris_training_file.csv', index=False)

    file_array = np.rot90(np.array([
        [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)]
    ]))
    testing_df = pd.DataFrame(file_array, columns=['image'])
    testing_df.to_csv('temp_data/solaris_inference_file.csv', index=False)


def mean_and_std_from_histogram(bins, cts):
    """Calculate the mean and standard deviation from a histogram."""

    bin_centers = bins[0:-1] + ((bins[1] - bins[0]) / 2.)
    mean = (np.sum(cts[0:] * bin_centers) / np.sum(cts[0:]) / 255)
    std = (np.sqrt((1. / sum(cts[0:])) * np.sum(cts[0:] * np.square(bin_centers - mean))) / 255)
    return mean, std


def get_tile_mean_sd(img_dir):
    """ function to calculate mean of images for preprocessing with yml file """

    # collect all image names
    ims = [f for f in os.listdir(img_dir)]

    # initialize array for each channel and create bins
    R_cts = np.zeros(shape=(51,), dtype='uint32')
    G_cts = np.zeros(shape=(51,), dtype='uint32')
    B_cts = np.zeros(shape=(51,), dtype='uint32')
    A_cts = np.zeros(shape=(51,), dtype='uint32')
    bins = np.arange(0, 260, 5)

    # loop through all images to calculate the histogram values
    for idx, im in enumerate(ims):
        curr_im = sol.utils.io.imread(os.path.join(img_dir, im))
        R_cts += np.array(np.histogram(curr_im[:, :, 0], bins=bins)[0], dtype='uint32')
        G_cts += np.array(np.histogram(curr_im[:, :, 1], bins=bins)[0], dtype='uint32')
        B_cts += np.array(np.histogram(curr_im[:, :, 2], bins=bins)[0], dtype='uint32')
        A_cts += np.array(np.histogram(curr_im[:, :, 3], bins=bins)[0], dtype='uint32')
        if idx % 100 == 0:
            print("# {} of {} completed".format(idx, len(ims)))

    # calculate mean and standard deviation for each channel
    r_mean, r_std = mean_and_std_from_histogram(bins, R_cts)
    g_mean, g_std = mean_and_std_from_histogram(bins, G_cts)
    b_mean, b_std = mean_and_std_from_histogram(bins, B_cts)
    a_mean, a_std = mean_and_std_from_histogram(bins, A_cts)

    scaled_mean = [r_mean, g_mean, b_mean, a_mean]
    scaled_sd = [r_std, g_std, b_std, a_std]

    # update the yml file with new values
    with open('temp_data/xdxd_spacenet4.yml') as f:
        update_config = yaml.load(f, Loader=yaml.FullLoader)

    for i in range(3):
        update_config['training_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])
        update_config['validation_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])
        update_config['inference_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])

        update_config['training_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])
        update_config['validation_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])
        update_config['inference_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])

    with open('temp_data/xdxd_spacenet4.yml', 'w') as f:
        update_config = yaml.dump(update_config, f)

    # Return the mean for images
    return (scaled_mean, scaled_sd)


def display_train_sample(sample_image, threshold=0.7):
    # import the images
    actual_image = skimage.io.imread(os.path.join('temp_data\\%s\\tile-256' % train_collection, sample_image))
    actual_mask = skimage.io.imread(os.path.join('temp_data\\%s\\mask-256' % train_collection, sample_image))
    continuous_pred = skimage.io.imread(os.path.join('temp_data\\%s\\inference_out' % train_collection, sample_image))
    continuous_pred = continuous_pred - continuous_pred.min()
    continuous_pred = continuous_pred / continuous_pred.max()

    # calculate the binary prediction
    inferred_vectors = sol.vector.mask.mask_to_poly_geojson(continuous_pred, bg_threshold=threshold)
    src_img_path = os.path.join('temp_data\\%s\\tile-256' % train_collection, sample_image)
    im_arr = skimage.io.imread(os.path.join('temp_data\\%s\\tile-256' % train_collection, sample_image))
    binary_pred = sol.vector.mask.footprint_mask(inferred_vectors, reference_im=src_img_path)

    # display sample results
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(actual_image, cmap='gray')
    axs[0, 0].set_title('Original Image Tile')
    axs[0, 1].imshow(actual_mask, cmap='gray')
    axs[0, 1].set_title('Actual Image Mask')
    axs[1, 0].imshow(continuous_pred, cmap='gray')
    axs[1, 0].set_title('Continous Probability Prediction')
    axs[1, 1].imshow(binary_pred, cmap='gray')
    axs[1, 1].set_title('Binary Prediction')
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.show()

def avg_jaccard_index(train_collection):
    scores = []
    ims = [f for f in os.listdir('temp_data\\%s\\tile-256' % train_collection)]
    for im in ims:
        actual_mask = skimage.io.imread(os.path.join('temp_data\\%s\\mask-256' % train_collection, im))
        continuous_pred = skimage.io.imread(os.path.join('temp_data\\%s\\inference_out' % train_collection, im))
        continuous_pred = continuous_pred - continuous_pred.min()
        continuous_pred = continuous_pred / continuous_pred.max()
        inferred_vectors = sol.vector.mask.mask_to_poly_geojson(continuous_pred, bg_threshold=0.7)
        src_img_path = os.path.join('temp_data\\%s\\tile-256' % train_collection, im)
        im_arr = skimage.io.imread(os.path.join('temp_data\\%s\\tile-256' % train_collection, im))
        binary_pred = sol.vector.mask.footprint_mask(inferred_vectors, reference_im=src_img_path)
        scores.append(jaccard_score(actual_mask, binary_pred, average='micro'))
    avg_jaccard = sum(scores)/len(scores)
    scores = np.array(scores)
    plt.hist(scores, bins=10)
    plt.title('Jaccard Index Histogram\n' + 'Average = ' + str(round(avg_jaccard,3)))
    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.show()

    return avg_jaccard

if __name__ == '__main__':
    use_project_path()
    train_collection = 'sample'
    test_collection = 'sample/tile-256'

    # Update the mean and standard dev pixel value in the yml file
    image_path = 'temp_data/%s/tile-256' % train_collection
    avg_pix, sd_pix = get_tile_mean_sd(image_path)
    print(avg_pix)
    print(sd_pix)

    # Create the csv files for training and making predictions
    create_train_csv(train_collection, test_collection)

    # Parse the yaml file into a dictionary
    config = sol.utils.config.parse('config/xdxd_spacenet4.yml')

    # Create the trainer and then kick off the training according to the config file settings
    # skip this if not training
    trainer = sol.nets.train.Trainer(config)
    trainer.train()

    # make predictions using the model
    inferer = sol.nets.infer.Inferer(config)
    inf_df = sol.nets.infer.get_infer_df(config)
    inferer(inf_df)

    # display a sample image from the training set
    display_train_sample('nia_825a50_104_265061_242184_19.png', threshold=.70)

    avg_score = avg_jaccard_index(train_collection)
    print(round(avg_score,3))