import tensorflow as tf
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import History
from lib.KerasHelpers.modelhelpers import model_placement
from src.models.model import TextCNN
from argparse import ArgumentParser
import deepdish as dd
import numpy as np
import os

DATA_DIR = './data/processed'

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
    
    parser.add_argument('--num_epochs', type=float, dest='num_epochs', 
                        help='number of epochs for training',
                        default=25)
    
    parser.add_argument('--dropout_rate', type=float, dest='dropout_rate', 
                        help='dropout rate for flatten conv outputs',
                        default=0.4)
    
    parser.add_argument('--batch_size', type=int, dest='batch_size', 
                        help='batch_Size for training',
                        default=32)
    
    parser.add_argument('--mode', dest='mode',
                        help='rand, static or nonstatic', default='rand')
    
    parser.add_argument('--debug', dest='debug', action = 'store_true',
                        help='debug mode', default=False)
    
    return parser

def check_opts(opts):
    assert opts.num_gpus >= 0
    assert opts.dataset in ['rt-polarity', '20_newsgroup']
    assert opts.lr > 0
    assert opts.batch_size > 0
    assert opts.debug in [True, False]
    
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    
    dataset_DIR = os.path.join(DATA_DIR, options.dataset)
    
    data_dict = dd.io.load(os.path.join(dataset_DIR, 'data.h5'))
    
    x_train, y_train, x_test, y_test = data_dict['x_train'], data_dict['y_train'], data_dict['x_test'], data_dict['y_test']
    
    validation_data = (x_test, y_test)
    
    kwargs = {'MAX_SEQUENCE_LENGTH': x_train.shape[1],
              'num_classes': y_train.shape[1],
              'num_words': data_dict['num_words'],
              'dropout_rate': options.dropout_rate,
              'flag': options.mode}
    
    if 'rand' in options.mode:
        kwargs['embedding_weights'] = None
    
    else:
        fname_wordvec = 'glove_'
        
        if options.wordvectors:
            fname_wordvec = 'word2vec_'
        
        kwargs['embedding_weights'] = np.load(os.path.join(dataset_DIR, fname_wordvec+'embedding.npy'))
    
              
    text_model = TextCNN(**kwargs)
    print (text_model.summary())
    model = model_placement(text_model, num_gpus=options.num_gpus)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=options.lr))
    
    num_examples = range(x_train.shape[0])
    history = History()
    callbacks = [history]
    num_epochs = options.num_epochs
              
    if options.debug:
        validation_data = None
        num_examples = range(1000)
        callbacks = None
        num_epochs = 5
    
    train_data = (x_train[num_examples], y_train[num_examples])
    
    model.fit(x=x_train[num_examples], y=y_train[num_examples], batch_size=options.batch_size,
              callbacks=callbacks, epochs=num_epochs, validation_data=validation_data)

if __name__ == '__main__':
    main()
              
     
