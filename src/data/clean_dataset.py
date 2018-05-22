from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import os
import sys


TEXT_DATA_DIR = '../../data/raw/RottenTomatoes/'

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

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
            texts.extend(t.split('\n'))
            f.close()
            labels.append(label_id)

print('Found %s texts.' % len(texts))

#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector

#print ('Longest %d sentence.' % max_len)

lengths = list(map(len, texts))

MAX_NB_WORDS = max(lengths)
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)