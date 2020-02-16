# AWS SageMaker Instructions
----

Getting running on SageMaker, via terminal...

Running Solaris on a SageMaker GPU instance requires creating a custom conda environment for solaris.  This will need to be done from a terminal window.  Once it has been completed, any Jupyter Notebooks requiring Solaris will need to be started with that environment.

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
pip install --user ipykernel (? not sure if needed)
python -m ipykernel install --user --name=persistent_solaris (? not sure if needed)
```

When starting a shell in the next instance, you can get the environment via:

```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate /home/ec2-user/SageMaker/DSC672/venv
```


```
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate tensorflow_p36
conda install -c conda-forge scikit-image
```



