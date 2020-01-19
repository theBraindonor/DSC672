# Segmenting Buildings for Disaster Resilience
## DSC 672 - Masters in Data Science Capstone Project
Authors:
* John Hoff
* Brian Pirlot
* Sebastian Zdarowski

## Initial Configuration

This project is configured to make use of two different Python virtual environments: one for GPU-enabled learning and one for CPU-only learning.  Windows 10 and Python version 3.7.4 were used to create the configuration and instructions.

_Please note that will this project is in active development the contents of the requirements(\_cpu).txt files will change frequently and need to be reloaded._

### GPU-Enabled Learning

```
> python -m virtualenv venv
> venv\scripts\activate
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/GDAL-3.0.3-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Fiona-1.8.13-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/Rtree-0.9.3-cp37-cp37m-win_amd64.whl
> pip install https://download.lfd.uci.edu/pythonlibs/q4hpdf1k/rasterio-1.1.2-cp37-cp37m-win_amd64.whl
> pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
(venv) > pip install -r requirements.txt
> pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

_note: everything above may need switched to anaconda_

### CPU-Only Learning

```
> python -m virtualenv venv_cpu
> venv_cpu\scripts\activate
(venv_cpu) > pip install -r requirements.txt
```

## Downloading Project Data
The data used for this project is too large to be included directly in this repository.  It is hosted in Amazon S3 by Driven Data.  The following files should be downloaded and extracted into the `raw_source_data` directory.

* [https://drivendata-public-assets.s3.amazonaws.com/train_tier_1.tgz](https://drivendata-public-assets.s3.amazonaws.com/train_tier_1.tgz) (32 GB)
* [https://drivendata-public-assets.s3.amazonaws.com/train_tier_2.tgz](https://drivendata-public-assets.s3.amazonaws.com/train_tier_2.tgz) (40 GB)
* [https://drivendata-public-assets.s3.amazonaws.com/test.tgz](https://drivendata-public-assets.s3.amazonaws.com/test.tgz) (9 GB)

Once that is done, the following directories should now be present in the `raw_source_data` directory:

* `train_tier_1`
* `train_tier_2`
* `test`
