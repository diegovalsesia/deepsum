# [DeepSUM: Deep neural network for Super-resolution of Unregistered Multitemporal images](https://arxiv.org/abs/1907.06490)

DeepSUM is a novel Multi Image Super-Resolution (MISR) deep neural network that exploits both spatial and temporal correlations to recover a single high resolution image from multiple unregistered low resolution images.

This repository contains python/tensorflow implementation of DeepSUM, trained and tested on the PROBA-V dataset provided by ESA’s [Advanced Concepts Team](http://www.esa.int/gsp/ACT/index.html) in the context of the [European Space Agency's Kelvin competition](https://kelvins.esa.int/proba-v-super-resolution/home/). 

DeepSUM is the winner of the PROBA-V SR challenge.


BibTex reference:
```
@ARTICLE{2019arXiv190706490B,
       author = {{Bordone Molini}, Andrea and {Valsesia}, Diego and {Fracastoro}, Giulia and
         {Magli}, Enrico},
        title = "{DeepSUM: Deep neural network for Super-resolution of Unregistered Multitemporal images}",
      journal = {arXiv e-prints},
     keywords = {Electrical Engineering and Systems Science - Image and Video Processing, Computer Science - Machine Learning},
         year = "2019",
        month = "Jul",
          eid = {arXiv:1907.06490},
        pages = {arXiv:1907.06490},
archivePrefix = {arXiv},
       eprint = {1907.06490},
 primaryClass = {eess.IV}
}
```

#### Setup to get started
Make sure you have Python3 and all the required python packages installed:
```
pip install -r requirements.txt
```


#### Load data from Kelvin Competition and create the training set and the validation set
- Download the PROBA-V dataset from the [Kelvin Competition](https://kelvins.esa.int/proba-v-super-resolution/data/) and save it under _./dataset\_creation/probav\_data_
- Load the dataset from the directories and save it to pickles by running _Save\_dataset\_pickles.ipynb_ notebook
- Run the _Create\_dataset.ipynb_ notebook to create training dataset and validation dataset for both bands NIR and RED
- To save RAM memory we advise to extract the best 9 images based on the masks: run _Save\_best9\_from\_dataset.ipynb_ notebook after _Create\_dataset.ipynb_. Based on the dataset you want to use (full or best 9) change the 'full' parameter in the config file.

#### Usage
In _config\_files/_ you can place your configuration before starting training the model:

```
"lr" : learning rate
"batch_size" batch size
"skip_step": validation frequency,
"dataset_path": directory with training set and validation set created by means of Create_dataset.ipynb,
"n_chunks": number of pickles in which the training set is divided,
"channels": number of channels of input images,
"T_in": number of images per scene,
"R": upscale factor,
"full": use the full dataset with all images or the best 9 for each imageset,
"patch_size_HR": size of input images,
"border": border size to take into account shifts in the loss and psnr computation,
"spectral_band": NIR or RED,
"RegNet_pretrain_dir": directory with RegNet pretraining checkpoint,
"SISRNet_pretrain_dir": directory with SISRNet pretraining checkpoint,
```

Run _DeepSUM\_train.ipynb_ to train a MISR model on the training dataset just generated. If _tensorboard\_dir_ directory is found in _checkpoints/_, the training will start from the latest checkpoint, otherwise the RegNet and SISRNet weights will be initialized from the checkpoints contained in the _pretraining\_checkpoints/_ directory. These weights come from the pretraining procedure explained in [DeepSUM paper](https://arxiv.org/abs/1907.06490).

#### Challenge checkpoints
The DeepSUM has been trained for both NIR and RED bands. In the 'checkpoints' directory there are the final weights used to produce the superresolved test images for the final ESA challenge submission.

_DeepSUM\_NIR\_lr\_5e-06\_bsize\_8_

_DeepSUM\_NIRpretraining\_RED\_lr\_5e-06\_bsize\_8_


#### Validation
During training, only the best 9 images for each imageset are considered for the score. After the training procedure is completed, you can compute a final evaluation on the validation set by also exploiting the other images available in each imageset. To do so, run _Sliding\_window\_evaluation.ipynb_.

#### Testing
- Run the _Create_testset.ipynb_ notebook under _dataset\_creation/_ to create the dataset with the test LR images
- To test the trained model on new LR images and get the corresponding superresolved images run _DeepSUM\_superresolve\_testdata.ipynb_.

## Authors & Contacts

DeepSUM is based on work by team *SuperPip* from the [Image Processing and Learning](https://ipl.polito.it/) group of Politecnico di Torino: Andrea Bordone Molini (andrea.bordone AT polito.it), Diego Valsesia (diego.valsesia AT polito.it), Giulia Fracastoro (giulia.fracastoro AT polito.it), Enrico Magli (enrico.magli AT polito.it).

