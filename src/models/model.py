from tensorflow.python.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Concatenate, Flatten, Dropout
from tensorflow.python.keras.models import Model

def TextCNN(input_shape, num_classes, vocab_size, embedding_layer, dropout_rate):

    """
    Input:
        - input_shape: maximum length of sentences
        - num_classes: number of classes in dataset
        - vocab_size: size of vocabulary
        - embedding_layer: embedding layer of Keras created by model type and static flags
        - dropout_rate: dropout rate for flattened pooled outputs

    Returns:
        - model: Model class created with specified inputs
    """


    x_input = Input(shape=(input_shape,), dtype='int32')

    #if 'rand' in flag:
    #    embedding_layer = Embedding(input_dim=vocab_size+1,
    #                                output_dim=EMBEDDING_DIM,
    #                                embeddings_initializer='uniform',
    #                                input_length=input_shape)
    #else:
    #    embedding_layer = Embedding(input_dim=vocab_size + 1,
    #                                output_dim=EMBEDDING_DIM,
    #                                weights=[],
    #                                input_length=input_shape,
    #                                trainable=static)

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

        pool = MaxPooling1D(pool_size=input_shape - kernel + 1)(conv)

        pooled.append(pool)

    merged = Concatenate()(pooled)

    flatten = Flatten()(merged)

    drop = Dropout(rate=dropout_rate)(flatten)

    x_output = Dense(num_classes, activation='sofmax', kernel_initializer='he_uniform')(drop)

    return Model(inputs=x_input, outputs=x_output)



