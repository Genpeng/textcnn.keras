# TextCNN.keras

Keras Implementation of "Convolutional Neural Networks for Sentence Classification" by Yoon Kim (2014)

## Requirements

Code is written in Python (3.5) and requires TensorFlow (1.7) and Keras (2.1.6)

Pre-trained word2vec vectors can be downloaded from [here](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz). Using the wget command. 

## Flags

CNN type - random, frozen word2vec, word2vec 

Training - GPU/CPU

Number of GPUs

Datasets 

## TO DO
* ~~Update clean_data.py to be generic script that can be applied to multiple datasets and saves in same format~~

* wget scripts in install.sh to download word vector files

* ~~Update requirements.txt accordingly~~

* ~~Flags for word2vec or glove vectors~~

* ~~Clean Data additions to script might be needed. Need to examine prepare data script for original paper.~~

* ~~Better handling for unknown word vectors, especially if trainable flag is False.~~ 
