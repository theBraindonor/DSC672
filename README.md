# Segmenting Buildings for Disaster Resilience
## DSC 672 - Masters in Data Science Capstone Project
Authors:
* John Hoff
* Brian Pirlot
* Sebastian Zdarowski

## Initial Configuration

This project makes use of two independant Python virtual environments. One for [Solaris](https://github.com/CosmiQ/solaris) and one for [Mask R-CNN](https://github.com/matterport/Mask_RCNN).  These environments have not been unified due to the steps necessary to install Solaris.  The package and some of its dependencies are new enough that they are not yet fully stable in pip and conda.

### Windows Installation

#### Solaris

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

#### Mask R-CNN

Be sure to use the `mcrnn-requirements-gpu.txt` file to make use of a local GPU.

```
> python -m virtualenv mrcnn_venv
> venv\scripts\activate
> pip install -r mrcnn-requirements.txt
```

### OSX Installation

#### Solaris

OSX installation does make use of both conda and pip.  Both GDAL and pytorch are unfortunately not fully working with solaris on the versions that will be installed by default.  For instance, you need gdal==3.0.2 to install solaris, but need gdal==2.3.3 in order to get everything to _run_.

```
> conda create --prefix ./venv python=3.7
> conda activate ./venv
> conda install gdal
> conda install pytorch torchvision -c pytorch
> pip install solaris
> conda install gdal==2.3.3
> conda install pytorch torchvision -c pytorch-nightly
> pip install pystac
> pip install rio-tiler
```

#### Mask R-CNN

```
> python -m virtualenv mrcnn_venv
> pip install -r mrcnn-requirements.txt
> venv\scripts\activate
```

### Linux Instalation

Please refer to the [SageMaker Guide](SageMaker.md) for details on installing the required Python modules for linux.

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

## Real Data Preparation

Once everything has been downloaded and extracted to the indicated folders, the training data can be created using the following commands:

```
> python -m prepare.extract_training_images -ts raw_source_data/train_tier_1 -ds temp_data/tier1 -s 256 -z 19
> python -m prepare.extract_training_images -ts raw_source_data/train_tier_1 -ds temp_data/tier1_lg -s 256 -z 20
> python -m prepare.extract_training_images -ts raw_source_data/train_tier_2 -ds temp_data/tier2 -s 256 -z 19
> python -m prepare.extract_training_images -ts raw_source_data/train_tier_2 -ds temp_data/tier2_lg -s 256 -z 20
```

This will create 256x256 pixel tiles at a zoom level of 19 and 20.

The testing data can be prepared using the following command:

```
> python -m prepare.resize_test_images -ts raw_source_data/test -ds temp_data/test -s 256
```

_Please note: Preparing the data will take a significant amount of time._

## License and Attribution

The following project includes source code and examples from the following projects:

### Solaris

### Mask R-CNN
