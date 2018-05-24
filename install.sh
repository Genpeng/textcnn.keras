#!/bin/bash

project_name

echo "~ C1 Secret ~"

echo "Creating Virtual Env"
virtualenv -p python3 env

echo "Activating Virtual Env"
source ./env/bin/activate

echo "Installing Requirements"
pip install -r requirements.txt

echo "Adding repos to lib"
mkdir lib
cd lib
git clone https://github.com/anmolsjoshi/KerasHelpers.git
cp KerasHelpers/__init__.py .
cd .

mkdir data/raw/wordvectors
cd data/raw/wordvectors


ipython kernel install --user --name=textCNN.keras

