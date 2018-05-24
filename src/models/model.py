from tensorflow.python.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout, Dense
from tensorflow.python.keras.models import Model
import numpy as np

def TextCNN(MAX_SEQUENCE_LENGTH, num_classes, num_words, dropout_rate=0.4, flag='rand', embedding_weights=None):

    """
    Input:
        - input_shape: maximum length of sentences
        - num_classes: number of classes in dataset
        - num_words: size of vocabulary
        - embedding_layer: embedding layer of Keras created by model type and static flags
        - dropout_rate: dropout rate for flattened pooled outputs

    Returns:
        - model: Model class created with specified inputs
    """
    EMBEDDING_DIM = 300
        
    x_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    if 'rand' in flag:
        embedding_layer = Embedding(input_dim=num_words,
                                    output_dim=EMBEDDING_DIM,
                                    embeddings_initializer='uniform',
                                    input_length=MAX_SEQUENCE_LENGTH)
        
    else:
        trainable_flag = False
        
        if 'non' in flag: trainable_flag = True
        
        embedding_layer = Embedding(input_dim=num_words,
                                    output_dim=EMBEDDING_DIM,
                                    weights=[embedding_weights],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=trainable_flag)

    x = embedding_layer(x_input)

    kernel_sizes = [3, 4, 5]
    pooled = []

    for kernel in kernel_sizes:

        conv = Conv1D(filters=100,
                      kernel_size=kernel,
                      padding='valid',
                      strides=1,
		      kernel_initializer='he_uniform',
                      activation='relu')(x)

        pool = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - kernel + 1)(conv)

        pooled.append(pool)

    merged = Concatenate(axis=1)(pooled)

    flatten = Flatten()(merged)

    drop = Dropout(rate=dropout_rate)(flatten)
   # dennse = Dense(300, activation='relu')(flatten)
    
    x_output = Dense(num_classes, kernel_initializer='he_uniform', activation='softmax')(drop)

    return Model(inputs=x_input, outputs=x_output)


if __name__ == '__main__':
    num_words = 20000
    EMBEDDING_DIM = 300
    embedding_weights = np.random.rand(num_words, EMBEDDING_DIM)
    TextCNN(MAX_SEQUENCE_LENGTH=1000, num_classes=2, num_words=20000).summary()
