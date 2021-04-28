import pandas as pd
import random
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
    return np.array(padded_sequences, dtype=object)
