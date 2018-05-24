#!/bin/bash

project_name

echo "~ C1 Secret ~"

echo "Creating Virtual Env"
virtualenv -p python3 env

echo "Activating Virtual Env"
source ./env/bin/activate

echo "Installing Requirements"
pip install -r requirements.txt

echo "Creating Jupyter Notebook Kernel"
ipython kernel install --user --name=textCNN.keras

echo "Adding repos to lib"
mkdir lib
cd lib
git clone https://github.com/anmolsjoshi/KerasHelpers.git
cp KerasHelpers/__init__.py .
cd ..

echo "Downloading word vectors"
mkdir data/raw/wordvectors
cd data/raw/wordvectors
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gunzip GoogleNews-vectors-negative300.bin.gz
echo "GoogleNews word2vec vectors downloaded"
wget -c "http://nlp.stanford.edu/data/glove.6B.zip"
mkdir glove.6B
unzip glove.6B.zip -d glove.6B/
rm -rf glove.6B.zip
echo "Glove vectors downloaded"
cd ..

echo "Downloading 20_newsgroup dataset"
wget -c "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz"
tar -xzf news20.tar.gz
rm -rf new20.tar.gz

echo "Downloading rt-polarity dataset"
mkdir rt-polarity
cd rt-polarity
wget -c "https://github.com/yoonkim/CNN_sentence/blob/master/rt-polarity.neg"
wget -c "https://github.com/yoonkim/CNN_sentence/blob/master/rt-polarity.pos"
cd ..
cd ..

echo "Setup Complete"
