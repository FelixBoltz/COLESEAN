from senticnet.senticnet import SenticNet
import pandas as pd
from sklearn.model_selection import train_test_split
# keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

# not needed right now
# import nltk
# import numpy as np
# import re
# import string

# testing the SenticNet python package here
sn = SenticNet()
concept_info = sn.concept('love')
polarity_label = sn.polarity_label('love')
polarity_value = sn.polarity_value('love')
moodtags = sn.moodtags('love')
semantics = sn.semantics('love')
sentics = sn.sentics('love')

# uncomment to see returned values
# print("Hello COLESEAN!", concept_info, polarity_label, polarity_value, moodtags, semantics, sentics)

# reading csv file to access reviews
print('Loading entire comments set:')
# index column of the csv-file seems to contain a string?
reviews = pd.read_csv('udemy_evaluate_latest.csv', dtype={0: "string", "learner_comment": "string"})
print('Found', len(reviews.index), 'comments')

# comments = reviews["learner_comment"]
# take a look at some comments
# for x in range(1):
#    print(comments[x])


# tokenize the comments
# tokens = comments.apply(nltk.word_tokenize)
# take a look at some tokenized comments
# print(tokens[:10])

# use keras to split up data into training and testing parts
reviews["learner_comment"].fillna("", inplace=True)
comments = reviews["learner_comment"].values
y = reviews["learner_rating"].values
# split data
sentences_train, sentences_test, y_train, y_test = train_test_split(comments, y, test_size=0.25, random_state=1000)
# tokenize data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
x_train = tokenizer.texts_to_sequences(sentences_train)
x_test = tokenizer.texts_to_sequences(sentences_test)
# discovered vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# define maximum allowed sequence size and use padding accordingly
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
