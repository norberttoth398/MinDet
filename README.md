# MinDet

[![Documentation Status](https://readthedocs.org/projects/mindet/badge/?version=latest)](https://mindet.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10061725.svg)](https://doi.org/10.5281/zenodo.10061725) [![Static Badge](https://img.shields.io/badge/mindet-interactive-notebook?link=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fnorberttoth%2Fmindet-interactive-notebook)](https://www.kaggle.com/code/norberttoth/mindet-interactive-notebook)




Repository for Deep Learning based petrography of igneous Plagioclase crystals based on circular polarised light images of thin sections. We make extensive use of the MMDetection library, with the work based on DetectoRS models.

## Install

In order to install this private package you must be able to access it (which you can if you're reading this) and run have/create a python 3.7 environment for relevant package requirements (PyTorch can be a pain like that). 

Ensure GCC is installed on your system.

#### Step 1
Create and activate environment:
	
	eg conda create -n MinDetEnv python=3.7 
	
	eg conda activate MinDetEnv

#### Step 2
Install required libraries (cluster nodes):

        wget https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp37-cp37m-linux_x86_64.whl

	pip install torch-1.7.0+cu110-cp37-cp37m-linux_x86_64.whl

	pip install torchvision==0.8.0 torchaudio==0.7.0

        pip install openmim

        pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

        mim install "mmdet<3.0.0"


Install required libraries (non-cluster):

	pip install torch==1.7.0+cu110 torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
	
	OR pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 (mac users)

	pip install openmim

	pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

	OR pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html (mac users)

	mim install "mmdet<3.0.0"

#### Install MinDet
Install using the following command: 

	pip install git+https://git@github.com/norberttoth398/MinDet


#### Supplementary Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10061710.svg)](https://doi.org/10.5281/zenodo.10061710)



#### Cite
Publication for the above repository is formally in the preparation stage. This work is presented at the EGU 2023 conference however, until formal publication of a manuscript please cite:

	norberttoth398. (2023). norberttoth398/MinDet:
