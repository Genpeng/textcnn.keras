import tensorflow as tf
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import History
from lib.KerasHelpers.modelhelpers import model_placement
from src.models.model import TextCNN
from argparse import ArgumentParser
import os


def build_parser():
    
    parser = ArgumentParser()

    parser.add_argument('--word2vec', action='store_true',
                        dest='wordvectors', help='Use word2vec vectors. No flag uses glove vectors.')

    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus', help='number of gpus. 0 for cpu only.',
                        metavar='num_gpus', default=0)

    parser.add_argument('--dataset', type=str, dest='dataset', 
                        help='dataset to use. rt-polarity, 20_newsgroup',
                        default='rt-polarity')
    
    parser.add_argument('--lr', type=float, dest='lr', 
                        help='learning rate for training',
                        default=0.001)
    
    parser.add_argument('--batch_size', type=int, dest='batch_size', 
                        help='batch_Size for training',
                        default=32)
    
    parser.add_argument('--debug', dest='debug', action = 'store_true',
                        help='debug mode', default=False)
    
    return parser

def check_opts(opts):
    assert opts.activation in ['sigmoid', 'tanh', 'softsign']
    assert opts.num_gpus >= 0 #and opts.num_gpus <= max_gpus
    assert opts.five_layer in [True, False]
    assert opts.normalization in [True, False]
    assert opts.dataset in ['mnist', 'cifar10', 'shapeset']
    assert opts.lr > 0
    assert opts.batch_size > 0
    assert opts.debug in [True, False]