import pandas as pd
import random
import pickle
import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences

reviews_file = 'comments_with_ratings.csv'


def get_reviews():
    # reading csv file to access reviews
    print('Loading entire comments set:')
    reviews = pd.read_csv(reviews_file)
    print('Found', len(reviews.iloc[:, 0]), 'comments')
    print('Found', len(reviews.iloc[:, 1]), 'ratings')
    return reviews


def even_out_distribution(ratings):
    min_rating_count = len(ratings)
    tier_indices = []
    for i in range(10):
        tier_indices.append([])

    for i in range(len(ratings)):
        if ratings[i] == 0.0:
            print('Error, did not expect a rating of 0.0')
        elif ratings[i] == 0.5:
            tier_indices[0].append(i)
        elif ratings[i] == 1.0:
            tier_indices[1].append(i)
        elif ratings[i] == 1.5:
            tier_indices[2].append(i)
        elif ratings[i] == 2.0:
            tier_indices[3].append(i)
        elif ratings[i] == 2.5:
            tier_indices[4].append(i)
        elif ratings[i] == 3.0:
            tier_indices[5].append(i)
        elif ratings[i] == 3.5:
            tier_indices[6].append(i)
        elif ratings[i] == 4.0:
            tier_indices[7].append(i)
        elif ratings[i] == 4.5:
            tier_indices[8].append(i)
        elif ratings[i] == 5.0:
            tier_indices[9].append(i)

    for i in range(len(tier_indices)):
        if len(tier_indices[i]) < min_rating_count:
            min_rating_count = len(tier_indices[i])
        # print("Bin:", i, "contains", len(tier_indices[i]), "entries")

    print("Lowest number of entries for a specific rating:", min_rating_count)

    for i in range(len(tier_indices)):
        random.seed(4)
        random.shuffle(tier_indices[i])

    for i in range(len(tier_indices)):
        diff = len(tier_indices[i]) - min_rating_count
        del tier_indices[i][:diff]

    even_distribution = []
    for i in range(len(tier_indices)):
        even_distribution.extend(tier_indices[i])
    random.seed(4)
    random.shuffle(even_distribution)
    print("Entries in evened out distribution:", len(even_distribution))
    return even_distribution


def get_we_sequences(tokenizer, sentences, max_len_we):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len_we)
    return np.asarray(padded_sequences).astype('float32')


def get_as_sequences(tokenizer, sentences, max_len_as, so):
    sequences = []
    for i in range(len(sentences)):
        # for i in range(10000):
        sequence = []
        concepts = so.search(sentences[i])
        for concept in concepts:
            sequence.append(tokenizer.word_index[concept])
        sequences.append(sequence)

    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len_as)
    return np.asarray(padded_sequences).astype('float32')


def get_polarity_scores(sentences, sn, so):
    concept_sequence = []
    for i in range(len(sentences)):
        # for i in range(10000):
        concepts = so.search(sentences[i])
        concept_sequence.append(concepts)
    sequence_polarities = []
    for i in range(len(concept_sequence)):
        comment_polarity = get_polarity(concept_sequence[i], sn)
        sequence_polarities.append(comment_polarity)
    return np.asarray(sequence_polarities).astype('float32')


def get_polarity(concept_list, sn):
    polarity_score = 0
    try:
        for i in range(len(concept_list)):
            polarity_score += float(sn.polarity_value(concept_list[i]))
    except KeyError:
        {}
    return polarity_score


def create_as_index(file):
    token2concepts = {}
    with open(file, 'r') as f:
        for line in f:
            concept = line.split(',')[0]
            tokens = concept.split('_')
            for token in tokens:
                if token not in token2concepts:
                    token2concepts[token] = set()
                token2concepts[token].add(concept)

    a_file = open("token2concepts.pkl", "wb")
    pickle.dump(token2concepts, a_file)
    a_file.close()


def create_sn_index(dictionary):
    token2concepts = {}
    for i in range(len(dictionary)):
        concept = list(dictionary.keys())[i]
        tokens = concept.split('_')
        for token in tokens:
            if token not in token2concepts:
                token2concepts[token] = set()
            token2concepts[token].add(concept)

    a_file = open("token2concepts.pkl", "wb")
    pickle.dump(token2concepts, a_file)
    a_file.close()


class SearchObject:

    def __init__(self):
        a_file = open("token2concepts.pkl", "rb")
        self.token2concepts = pickle.load(a_file)
        a_file.close()
        nltk.download('punkt')

    def search(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)

        found_concepts = []
        for t in tokens:
            if t in self.token2concepts:
                potential_concepts = sorted(self.token2concepts[t], key=len, reverse=True)
                for concept in potential_concepts:
                    if concept.replace('_', ' ') in text:
                        if not found_concepts:
                            found_concepts += [concept]
                        # if subsequent tokens activate the same concept, it must not be saved
                        elif found_concepts[-1] != concept:
                            found_concepts += [concept]
                        break
        return found_concepts


def create_embedding_matrix_glove(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def create_embedding_matrix_fasttext(model, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        try:
            vector = model[word]
            embedding_vector = np.array(vector, dtype=np.float32)[:embedding_dim]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix


def create_embedding_matrix_as(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split(",")
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
