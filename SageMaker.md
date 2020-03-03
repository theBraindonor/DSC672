# SageMaker Instructions

The recommended process for setting this project up on Amazon SageMaker is to first launch a new instance connected to this repository.  While it is possible to do this without connecting to a repository, this guide assumes that all project code resides in `~/SageMaker/DSC672`.  All training is then performed using the terminal.

These instructions will install the Python virtual environments into the SageMaker mount point and saved when the instance is stopped.

## Solaris Instructions

The Solaris configuration is used for all data extraction and initial analysis.  The configuration is also used for training and inferencing with the Solaris based model.

### Initial Configuration

This needs to be done once to create the virtual environment and save it to the SageMaker mount point.

```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
cd SageMaker
git clone https://github.com/cosmiq/solaris.git
cd solaris
conda env create -f environment-gpu.yml --prefix ../DSC672/venv
conda activate /home/ec2-user/SageMaker/DSC672/venv
pip install .
pip install pystac
pip install rio-tiler
pip install Pillow==6.1
```

### Restart Configuration

When an instance is restarted that has been configured for the Solaris model, the Python virtual environment can be activated by the following code.

```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate /home/ec2-user/SageMaker/DSC672/venv
```

## Mask R-CNN Instructions

The Mask R-CNN configuration is only used for training and inferencing with the Mask R-CNN based model.

### Initial Configuration

This following steps need to be taken when an instance is first started to configure it to build the Mask R-CNN model.

```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
cd SageMaker
conda activate tensorflow_p36
conda env export > environment.yml
conda env create -f environment.yml --prefix ./DSC672/mrcnn_venv
conda activate /home/ec2-user/SageMaker/DSC672/mrcnn_venv
conda install -c conda-forge scikit-image
conda install -c conda-forge scikit-learn
```

### Restart Configuration

When an instance is restarted that has been configured for the Mask R-CNN model, the Python virtual environment can be activated by the following code.

```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate /home/ec2-user/SageMaker/DSC672/mrcnn_venv
```
