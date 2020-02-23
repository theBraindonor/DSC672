
import solaris as sol
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import glob
import os
import time

from utility import use_project_path

import yaml # this needs to be installed for the environment
from sklearn.metrics import jaccard_score # this needs to be installed for the environment
from sklearn.model_selection import StratifiedShuffleSplit # this needs to be installed for the environment

def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    if inference_set == 'test':

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/%s/tile-256/**/*.png' % train_set, recursive=True)],
            [filename for filename in glob.iglob('temp_data/%s/mask-256/**/*.png' % train_set, recursive=True)],
        ]))

        training_df = pd.DataFrame(file_array, columns=['image', 'label'])
        training_df.to_csv('temp_data/solaris_training_file.csv', index=False)

        file_array = np.rot90(np.array([
            [filename for filename in glob.iglob('temp_data/temp_test/resize/*.png', recursive=True)],
        ]))

        testing_df = pd.DataFrame(file_array, columns=['image'])
        testing_df.to_csv('temp_data/solaris_inference_file.csv', index=False)

    else:

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


def sample_train_csv(tier1_pct, tier2_pct, random_state=4):
    """ function to create a training file by sampling given percentages from tier1 and tier2 """

    # build a dataframe with all tier1 file paths
    tier1_tile = np.array([filename for filename in glob.iglob('temp_data/tier1/tile-256/*.png', recursive=True)])
    tier1_mask = np.array([filename for filename in glob.iglob('temp_data/tier1/mask-256/*.png', recursive=True)])
    tier1_df = pd.DataFrame({'image': tier1_tile, 'label': tier1_mask})
    tier1_df['region'] = tier1_df['image'].map(lambda x: os.path.split(x)[1][0:3])

    # build a dataframe with all tier2 file paths
    tier2_tile = np.array([filename for filename in glob.iglob('temp_data/tier2/tile-256/*.png', recursive=True)])
    tier2_mask = np.array([filename for filename in glob.iglob('temp_data/tier2/mask-256/*.png', recursive=True)])
    tier2_df = pd.DataFrame({'image': tier2_tile, 'label': tier2_mask})
    tier2_df['region'] = tier2_df['image'].map(lambda x: os.path.split(x)[1][0:3])

    # create a stratified sample by region from tier1 and tier2
    split_tier1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - tier1_pct), random_state=random_state)
    split_tier2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - tier2_pct), random_state=random_state)

    for train_index, test_index in split_tier1.split(tier1_df, tier1_df['region']):
        strat_tier1_set = tier1_df.loc[train_index]

    for train_index, test_index in split_tier2.split(tier2_df, tier2_df['region']):
        strat_tier2_set = tier2_df.loc[train_index]

    # create the outfile by combining the results for tier1 and tier2
    train_out_df = strat_tier1_set.append(strat_tier2_set)
    train_out_df.drop(columns=['region'], inplace=True)
    train_out_df = train_out_df.sample(frac=1)  # shuffle the dataframe
    train_out_df.to_csv('temp_data/solaris_training_file.csv', index=False)

    # return list of images to calculating mean and std
    return train_out_df['image'].to_list()


def mean_and_std_from_histogram(bins, cts):
    """Calculate the mean and standard deviation from a histogram."""

    bin_centers = bins[0:-1] + ((bins[1] - bins[0]) / 2.)
    mean = np.sum(cts[0:] * bin_centers) / np.sum(cts[0:])
    std = np.sqrt((1. / sum(cts[0:])) * np.sum(cts[0:] * np.square(bin_centers - mean)))
    return mean, std


def get_tile_mean_sd(ims):
    """ function to calculate mean of images for preprocessing with yml file """

    # collect all image names
    #ims = [f for f in os.listdir(img_dir)]

    # initialize array for each channel and create bins
    R_cts = np.zeros(shape=(51,), dtype='uint32')
    G_cts = np.zeros(shape=(51,), dtype='uint32')
    B_cts = np.zeros(shape=(51,), dtype='uint32')
    A_cts = np.zeros(shape=(51,), dtype='uint32')
    bins = np.arange(0, 260, 5)

    # loop through all images to calculate the histogram values
    for idx, im in enumerate(ims):
        curr_im = sol.utils.io.imread(im)
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

    scaled_mean = [r_mean/255, g_mean/255, b_mean/255, a_mean/255]
    scaled_sd = [r_std/255, g_std/255, b_std/255, a_std/255]

    # update the yml file with new values
    with open('config/xdxd_spacenet4.yml') as f:
        update_config = yaml.load(f, Loader=yaml.FullLoader)

    for i in range(3):
        update_config['training_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])
        update_config['validation_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])
        update_config['inference_augmentation']['augmentations']['Normalize']['mean'][i] = float(scaled_mean [i])

        update_config['training_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])
        update_config['validation_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])
        update_config['inference_augmentation']['augmentations']['Normalize']['std'][i] = float(scaled_sd[i])

    with open('config/xdxd_spacenet4.yml', 'w') as f:
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
    axs[1, 1].set_title('Binary Prediction - ' + str(round(jaccard_score(actual_mask, binary_pred, average='micro'), 3)))
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
        inferred_vectors = sol.vector.mask.mask_to_poly_geojson(continuous_pred, bg_threshold=0.6)
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

    # Create a training file sampled from tier1 and tier2
    image_list = sample_train_csv(tier1_pct=0.6, tier2_pct=0.25, random_state=4)

    # Update the mean and standard dev pixel value in the yml file
    avg_pix, sd_pix = get_tile_mean_sd(image_list)
    print(avg_pix)
    print(sd_pix)

    # Parse the yaml file into a dictionary
    config = sol.utils.config.parse('config/xdxd_spacenet4.yml')

    # Create the trainer and then kick off the training according to the config file settings
    # skip this if not training
    trainer = sol.nets.train.Trainer(config)

    start = time.time()
    trainer.train()
    end = time.time()
    print('{} minutes'.format(round((end - start) / 60, 2)))

    # make predictions using the model
    #inferer = sol.nets.infer.Inferer(config)
    #inf_df = sol.nets.infer.get_infer_df(config)
    #inferer(inf_df)

    # display a sample image from the training set
    #display_train_sample('nia_825a50_107_265064_242184_19.png', threshold=.6)

    #avg_score = avg_jaccard_index(train_collection)