# input configuration
# model
# WordEmbedding
# SenticNet Inclusion
# output configuration

import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras import layers
from keras.layers import LSTM, Bidirectional


class Regressor:
    word_embedding = ''
    sentic_net = ''
    model_type = 0

    def __init__(self, word_embedding, sentic_net, model_type):
        self.word_embedding = word_embedding
        self.sentic_net = sentic_net
        self.model_type = model_type

    def train(self, input_data):
        return 0+input_data


def concept_vector_model(maxlen_we, maxlen_as):
    tf.keras.backend.clear_session()
    we_input = keras.Input(shape=(maxlen_we,), name="we_sequence")
    as_input = keras.Input(shape=(maxlen_as,), name="concept_sequence")
    we_embedding_layer = layers.Embedding(vocab_size_WE, embedding_dim_WE, weights=[embedding_matrix_WE],
                                          input_length=maxlen_WE, trainable=False)(we_input)
    as_embedding_layer = layers.Embedding(vocab_size_AS, embedding_dim_AS, weights=[embedding_matrix_AS],
                                          input_length=maxlen_AS, trainable=False)(as_input)
    we_BiLSTM = Bidirectional(LSTM(64))(we_embedding_layer)
    as_BiLSTM = Bidirectional(LSTM(64))(as_embedding_layer)
    we_dense = layers.Dense(128, activation='relu')(we_BiLSTM)
    as_dense = layers.Dense(128, activation='relu')(as_BiLSTM)
    conc = layers.concatenate([we_dense, as_dense])
    extra_dense = layers.Dense(10, activation='relu')(conc)
    output = layers.Dense(1)(extra_dense)
    model = keras.Model(inputs=[we_input, as_input], outputs=[output], )
    return model

# def polarity_vector_model():

# def polarity_score_model():
