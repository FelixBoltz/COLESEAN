from senticnet.senticnet import SenticNet
import pandas as pd
import nltk
import numpy as np
import re
import string

# testing the SenticNet python package
sn = SenticNet()
concept_info = sn.concept('love')
polarity_label = sn.polarity_label('love')
polarity_value = sn.polarity_value('love')
moodtags = sn.moodtags('love')
semantics = sn.semantics('love')
sentics = sn.sentics('love')

print("Hello COLESEAN!", concept_info, polarity_label, polarity_value, moodtags, semantics, sentics)

# reading csv file to access reviews
print('Loading entire comments set:')
# index column of the csv-file seems to contain a string?
reviews = pd.read_csv('udemy_evaluate_latest.csv', dtype={0: "string"})
print('Found', len(reviews.index), 'comments')

comments = reviews["learner_comment"]

# take a look at some comments
for x in range(1):
    print(comments[x])


# tokenize the comments
comments.fillna("", inplace=True)
tokens = comments.apply(nltk.word_tokenize)
# take a look at some tokenized comments
print(tokens[:10])

# need to clean the comments, maybe use keras.preprocessing instead of nltk?
# words = [word for word in tokens if word.isalpha()]
