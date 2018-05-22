from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
import deepdish as dd
import numpy as np
import pandas as pd
import os
import sys


TEXT_DATA_DIR = '../../data/raw/rt-polarity/'
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

print('Indexing word vectors.')

model = KeyedVectors.load_word2vec_format('../../data/raw/wordvectors/GoogleNews-vectors-negative300.bin', binary=True)

#embeddings_index = {}
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs

#print('Found %s word vectors.' % len(embeddings_index))

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
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
            split_text = t.split('\n')
            texts.extend(split_text)
            f.close()            
            labels.extend(len(split_text)*[label_id])

print('Found %s texts.' % len(texts))
print ('Found %s labels.' % len(labels))

#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector

#print ('Longest %d sentence.' % max_len)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in model.vocab:
        embedding_vector = model[word]

print (embedding_matrix.shape)

np.save('../../data/processed/rt-polarity/rt-embedding', embedding_matrix, allow_pickle=False)
dd.io.save('../../data/processed/rt-polarity/rt-processed-text.h5', {'data': data, 'labels': labels})
print ('Saved file')