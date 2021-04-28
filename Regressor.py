import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import LSTM, Bidirectional


def concept_vector_model(max_len_we, max_len_as, vocab_size_we, vocab_size_as,
                         embedding_dim_we, embedding_matrix_we,
                         embedding_dim_as, embedding_matrix_as):
    # defining keras model
    tf.keras.backend.clear_session()
    we_input = keras.Input(shape=(max_len_we,), name="we_sequence")
    as_input = keras.Input(shape=(max_len_as,), name="concept_sequence")
    we_embedding_layer = layers.Embedding(vocab_size_we, embedding_dim_we, weights=[embedding_matrix_we],
                                          input_length=max_len_we, trainable=False)(we_input)
    as_embedding_layer = layers.Embedding(vocab_size_as, embedding_dim_as, weights=[embedding_matrix_as],
                                          input_length=max_len_as, trainable=False)(as_input)
    we_bi_lstm = Bidirectional(LSTM(64))(we_embedding_layer)
    as_bi_lstm = Bidirectional(LSTM(64))(as_embedding_layer)
    we_dense = layers.Dense(128, activation='relu')(we_bi_lstm)
    as_dense = layers.Dense(128, activation='relu')(as_bi_lstm)
    concat = layers.concatenate([we_dense, as_dense])
    extra_dense = layers.Dense(10, activation='relu')(concat)
    output = layers.Dense(1)(extra_dense)
    model = keras.Model(inputs=[we_input, as_input], outputs=[output], )
    return model

# def polarity_vector_model():

# def polarity_score_model():
