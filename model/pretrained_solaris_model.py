import os
import glob
import csv
import solaris as sol
import skimage
import matplotlib.pyplot as plt
import numpy as np


def create_train_csv(train_set, inference_set):
    """
    Function to create the train CSV files for solaris.
    Options for train_set and inference_set include:
    sample, tier1, tier2, test
    The set names should be passed as a string with quotes
    """

    # This references the project director "DSC672"
    project_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))

    # Collect the absolute file paths for the image files
    os.chdir(project_dir + '\\temp_data\\' + train_set + '\\tile-256')
    tile_paths = []
    for dirpath,_,filenames in os.walk(os.getcwd()):
        for f in filenames:
            tile_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    tile_paths.sort()
    tile_paths.insert(0,'image')

    # Collect the absolute file paths for the mask files
    os.chdir(project_dir + '\\temp_data\\' + train_set + '\\mask-256')
    mask_paths = []
    for dirpath,_,filenames in os.walk(os.getcwd()):
        for f in filenames:
            mask_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    mask_paths.sort()
    mask_paths.insert(0,'label')

    # Create the csv files to the approriate temp_data subfolder
    with open(project_dir + '\\temp_data\\' + train_set + '\\training_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(tile_paths, mask_paths))

    with open(project_dir + '\\temp_data\\' + train_set + '\\inference_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(tile_paths))

# Create the csv files for training and making predictions
create_train_csv("sample", "sample")

# Parse the yaml file into a dictionary - will need to update this absolute path
# The yml file also contains absolute references that will need to be updated for each user
config = sol.utils.config.parse('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\xdxd_spacenet4.yml')

# Create the trainer and then kick off the training according to the config file settings
# skip this if not training
#trainer = sol.nets.train.Trainer(config)
#trainer.train()

# !!! This is where i'm a little lost on how to create predictions - This creates results but with warnings !!! #
inferer = sol.nets.infer.Inferer(config)
inf_df = sol.nets.infer.get_infer_df(config)
inferer(inf_df)

# create sample images to view results for 1 tile
actual_image_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\tile-256\\nia_825a50_107_265064_242184_19.png')
actual_mask_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\mask-256\\nia_825a50_107_265064_242184_19.png')
continuous_prediction_sample = skimage.io.imread('C:\\Users\\Brian\\Documents\\GitHub\\DSC672\\temp_data\\sample\\inference_out\\nia_825a50_107_265064_242184_19.png')
binary_prediction_sample = (np.digitize(continuous_prediction_sample, np.array([0, 128])) - 1)*255

# display sample results
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(actual_image_sample, cmap='gray')
axs[0, 0].set_title('Original Image Tile')
axs[0, 1].imshow(actual_mask_sample, cmap='gray')
axs[0, 1].set_title('Actual Image Mask')
axs[1, 0].imshow(continuous_prediction_sample, cmap='gray')
axs[1, 0].set_title('Continous Probability Prediction')
axs[1, 1].imshow(binary_prediction_sample, cmap='gray')
axs[1, 1].set_title('Binary Prediction')
plt.show()