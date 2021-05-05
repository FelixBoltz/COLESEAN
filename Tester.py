import Preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from senticnet.senticnet import SenticNet
import pandas as pd
import matplotlib.pyplot as plt

# Word Embedding variable, possible values: 'GloVe', 'W2V' and 'FastText'
import Regressor

word_embedding = 'GloVe'
# SenticNet variable, possible values: 'concept_vector', 'polarity_vector' and 'polarity_score'
sentic_net = 'concept_vector'
# model type, possible values: 0 (concept vector model), 1 (polarity vector model) and 2 (polarity score model)
model_type = 0
# GloVe vectors location
glove_path = 'glove.6B.300d.txt'
# AffectiveSpace vectors location
affectivespace_path = 'affectivespace.csv'
# loss function: either mse or mae
loss_function = 'mae'


def main():
    # input variables
    reviews = Preprocessing.get_reviews()
    comments = reviews.iloc[:, 1]
    ratings = reviews.iloc[:, 2]
    # even out the distribution
    even_distribution = Preprocessing.even_out_distribution(ratings)
    even_ratings = []
    even_comments = []
    for i in range(len(even_distribution)):
        even_ratings.append(ratings[even_distribution[i]])
        even_comments.append(comments[even_distribution[i]])
    # print("Ratings length:", len(even_ratings))
    # print("Comments length:", len(even_comments))

    # split data into 70% training, 10% validation and 20% test
    sentences_train_val, sentences_test, y_train_val, y_test = train_test_split(even_comments, even_ratings,
                                                                                train_size=0.8, test_size=0.2,
                                                                                random_state=1000)
    sentences_train, sentences_val, y_train, y_val = train_test_split(sentences_train_val, y_train_val,
                                                                      train_size=0.875, test_size=0.125,
                                                                      random_state=1000)
    y_train = Preprocessing.np.asarray(y_train).astype('float32')
    y_val = Preprocessing.np.asarray(y_val).astype('float32')
    y_test = Preprocessing.np.asarray(y_test).astype('float32')
    # set maximal sequence length for word embedding branch input
    max_len_we = 100
    # get padded token sequences for word embedding branch input
    tokenizer_we = Tokenizer()
    tokenizer_we.fit_on_texts(comments)
    vocab_size_we = len(tokenizer_we.word_index) + 1
    x_train_we_pad = Preprocessing.get_we_sequences(tokenizer_we, sentences_train, max_len_we)
    x_val_we_pad = Preprocessing.get_we_sequences(tokenizer_we, sentences_val, max_len_we)
    x_test_we_pad = Preprocessing.get_we_sequences(tokenizer_we, sentences_test, max_len_we)
    # set up embedding matrix for word embedding input
    embedding_dim_we = 300
    embedding_matrix_we = []
    if word_embedding == 'GloVe':
        embedding_matrix_we = Preprocessing.create_embedding_matrix_glove(glove_path,
                                                                          tokenizer_we.word_index, embedding_dim_we)
    # prepare SenticNet related input depending on model
    so = Preprocessing.SearchObject()
    if model_type == 0:
        Preprocessing.create_as_index(affectivespace_path)
        affectivespace = pd.read_csv(affectivespace_path, header=None, keep_default_na=False)
        tokenizer_as = Tokenizer()
        for i in range(len(affectivespace.index)):
            concept_name = affectivespace.loc[i, 0]
            concept_name = concept_name.lower()
            tokenizer_as.word_index[concept_name] = i

        vocab_size_as = len(tokenizer_as.word_index) + 1
        # set maximal sequence length for concept vector
        max_len_as = 100
        x_train_as_pad = Preprocessing.get_as_sequences(tokenizer_as, sentences_train, max_len_as, so)
        x_val_as_pad = Preprocessing.get_as_sequences(tokenizer_as, sentences_val, max_len_as, so)
        x_test_as_pad = Preprocessing.get_as_sequences(tokenizer_as, sentences_test, max_len_as, so)
        # set up embedding matrix for concept vector input
        embedding_dim_as = 100
        embedding_matrix_as = \
            Preprocessing.create_embedding_matrix_glove(affectivespace_path, tokenizer_as.word_index, embedding_dim_as)
        model = Regressor.concept_vector_model(max_len_we,
                                               max_len_as, vocab_size_we, vocab_size_as, embedding_dim_we,
                                               embedding_matrix_we, embedding_dim_as, embedding_matrix_as)
        model.compile(optimizer='Adam', loss=loss_function, metrics=['mae', 'mse'])
        model.summary()
        # early stopping conditions
        es = Regressor.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=3, verbose=1)
        history = model.fit({"we_sequence": x_train_we_pad, "concept_sequence": x_train_as_pad}, y_train,
                            epochs=100,
                            verbose=1,
                            validation_data=({"we_sequence": x_val_we_pad, "concept_sequence": x_val_as_pad}, y_val),
                            callbacks=[es],
                            batch_size=64)
        # training results
        results = model.evaluate({"we_sequence": x_train_we_pad, "concept_sequence": x_train_as_pad}, y_train,
                                 verbose=1)
        print("Training results: ", results)
        # show training graph
        plot_history(history)
        plt.show()
        Regressor.print_test_performance(model, y_test, x_test_we_pad, x_test_as_pad)
    elif model_type == 1:
        sn = SenticNet()
        Preprocessing.create_sn_index(sn.data)

    return 0


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 1, 1)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


if __name__ == "__main__":
    main()
