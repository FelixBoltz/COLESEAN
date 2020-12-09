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
print('Loading entire comments set')
reviews = pd.read_csv('udemy_evaluate_latest.csv')
print('Found', len(reviews.index), 'comments')

# for x in range(5):
#    print(reviews.values[x])

comments = reviews["learner_comment"]

# take a look at the first 10 comments
for x in range(5):
    print(comments[x])


# tokenize the comments
comments.fillna("", inplace=True)
print(comments.tolist()[:4])
tokens = comments.apply(nltk.word_tokenize)
comment0 = nltk.word_tokenize(comments[0])
print('First comment tokenized:', comment0)
print(tokens[:10])

# need to clean the comments
# words = [word for word in tokens if word.isalpha()]
