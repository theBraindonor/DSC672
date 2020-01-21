# Segmenting Buildings for Disaster Resilience
## DSC 672 - Masters in Data Science Capstone Project
Authors:
* John Hoff
* Brian Pirlot
* Sebastian Zdarowski

## Initial Configuration

Due to the nature of some of the python libraries, an easy `pip install -r requirements.txt` will not be able to set everything up.  Instead, multiple wheels will need to be manually installed before installing the requirements.  This has been tested on Windows using 64-bit Python 3.7.4.  Using other Python versions or operating systems will require significant changes to the configuration.

The following set of commands will manually install wheel files for the modules that will not work out of the box and allow for pip to process the requirements.txt file correctly.

```
> python -m virtualenv venv
> venv\scripts\activate
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/GDAL-3.0.3-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Fiona-1.8.13-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Rtree-0.9.3-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/rasterio-1.1.2-cp37-cp37m-win_amd64.whl
> pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
> pip install -r requirements.txt
```

The above commands currently only support gpu-based learning.  Additional cpu-based learning commands may become necessary.

## Sample Data Preparation

Some sample data is included in this repository so that configuration can be quickly checked _without_ having to download the full training and testing data files.  Simply run the following command from the Python virtual environment:

```
> python -m prepare.extract_training_images
```

The `prepare/extract_training_images.py` file can also be run within a Python IDE directly.

## Downloading Real Project Data
The data used for this project is too large to be included directly in this repository.  It is hosted in Amazon S3 by Driven Data.  The following files should be downloaded and extracted into the `raw_source_data` directory.

* [https://drivendata-public-assets.s3.amazonaws.com/train_tier_1.tgz](https://drivendata-public-assets.s3.amazonaws.com/train_tier_1.tgz) (32 GB)
* [https://drivendata-public-assets.s3.amazonaws.com/train_tier_2.tgz](https://drivendata-public-assets.s3.amazonaws.com/train_tier_2.tgz) (40 GB)
* [https://drivendata-public-assets.s3.amazonaws.com/test.tgz](https://drivendata-public-assets.s3.amazonaws.com/test.tgz) (9 GB)

Once that is done, the following directories should now be present in the `raw_source_data` directory:

* `train_tier_1`
* `train_tier_2`
* `test`
