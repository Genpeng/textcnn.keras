from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
import deepdish as dd
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import os, sys, re


MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

def build_parser():
    
    parser = ArgumentParser()
    
    parser.add_argument('--glove', action='store_true',
                        dest='wordvector', help='Flag to use glove word vectors instead of word2vec.',
                        default=False)

    parser.add_argument('--dataset', type=str, dest='dataset', 
                        help='dataset to use, rt-polarity, 20_newsgroup',
                        default='rt-polarity') 
    return parser

def check_opts(opts):
    assert opts.wordvector in [True, False]
    assert opts.dataset in ['rt-polarity', '20_newsgroup']
    
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def data_from_directory(directory):
    
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                if fname.isdigit():
                    texts.append(clean_str(t))
                    f.close()
                    labels.append(label_id)
                else:
                    split_text = t.split('\n')
                    texts.extend([clean_str(txt) for txt in split_text])
                    f.close()            
                    labels.extend(len(split_text)*[label_id])

    return texts, labels

def get_word_vectors(word_index, glove_flag=False):
    
    counter = 0
    
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    
    if glove_flag:
    
        embeddings_index = {}
        f = open(os.path.join('data/raw/wordvectors', 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                counter = counter + 1
    else:
        
        model = KeyedVectors.load_word2vec_format('data/raw/wordvectors/GoogleNews-vectors-negative300.bin', binary=True)

        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        for word, i in word_index.items():
            if word in model.vocab:
                embedding_matrix[i] = model[word]
                counter = counter + 1
    
    print ('Found %s word vectors.' % counter)
   
    return embedding_matrix
    

def main():
    
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    
    DATA_DIR = 'data/raw'
    TEXT_DATA_DIR = os.path.join(DATA_DIR, options.dataset)
    
    print('Processing text dataset')
    texts, labels = data_from_directory(TEXT_DATA_DIR)
    print('Found %s texts.' % len(texts))
    print ('Found %s labels.' % len(labels))
    
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    print('Preparing embedding matrix.')
    embedding_matrix = get_word_vectors(word_index, glove_flag=options.wordvector)
    print('Shape of embedding matrix:', embedding_matrix.shape)
    
    processed_DIR = os.path.join('data/processed', options.dataset)
    
    if not os.path.exists(processed_DIR):
        os.makedirs(processed_DIR)

    np.save(os.path.join(processed_DIR, 'embedding.npy'), embedding_matrix, allow_pickle=False)
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.10, random_state=42, shuffle=True)  
    dd.io.save(os.path.join(processed_DIR, 'data.h5'), {'x_train': x_train, 'y_train': y_train, 
                                                        'x_test': x_test, 'y_test': y_test})
    
    print ('Saved embedding.npy and data.h5 in %s' % processed_DIR)
    
if __name__ == '__main__':
    main()
