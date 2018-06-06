# TextCNN.keras

[Keras Implementation of "Convolutional Neural Networks for Sentence Classification" by Yoon Kim (2014)](https://arxiv.org/abs/1408.5882)

## Implementation Details
This repository contains the code of an implementation of Yoon Kim's paer on using CNNs for Sentence Classification. The code is written using Tensorflow/Keras. A few differences from the original implementation is the use of `He Initialization`, since `relu' activation is used.  

## Documentation

### Setup
Run `./install.sh` in the working directory. It installs all the requirements from `requirements.txt` into a virtual environment, and clones the `KerasHelpers` git repo in to lib/ folder. It downloads the [GoogleNews word2vec 300-dim vectors](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz) and [glove.6B](https://nlp.stanford.edu/projects/glove/) vectors to `data/raw/` folder. It also creates a jupyter kernel called `textcnn.keras` that allows jupyter notebooks to be created using the virtual environment. 

#### requirements.txt
* tensorflow-gpu
* scikit-learn
* deepdish
* gensim
* h5py==2.8.0rc1 (Until new h5py update)
* ipykernel

#### KerasHelpers
[KerasHelpers]("https://github.com/anmolsjoshi/KerasHelpers") is a repository containing helper functions for Keras experiments. 

### Running Experiments
Use train_model.py to train model on different datasets, hyperparameters and word embeddings. Run `python train_model.py` to conduct experiment. Training takes a few minutes on a GTX 1080 Ti. **Before you run this, you should run `./install.sh`**.

#### experiment.py
`train_model.py` At the end of training, history of the model is saved into `model_history/name_of_dataset/name_of_experiment/`:

* `history.h5` : h5py file containing a dictionary of the training and validation accuracy and loss. 

##### Example:
    python train_model.py --word2vec --dataset 20_newsgroup --num_gpus 1
          
##### Flags
* `--debug`: Debug mode. Runs model on reduced dataset for 5 epochs. Mode is used to ensure requirements and libraries are working correctly. Does not save history. 
* `--word2vec`: Flag to use GoogleNews word2vec vector. If not called, glove word vectors is used.
* `--dataset`: Dataset to train network on. rt-polarity, 20_newsgroup can be used. Default is rt-polarity. 
* `--lr`: Learning rate for model. Default 1e-3. 
* `--batch_size`: Batch size for model. Default is 32. If you use num_gpus 2, increase batch size for better performance. 
* `--num_epochs`: Number of epochs for model fitting. Default 25. 
* `--num_gpus`: Number of GPUs. Greater than 1 will run in parallel mode. 1 will use GPU. 0 will use CPU. Default is 0. 
