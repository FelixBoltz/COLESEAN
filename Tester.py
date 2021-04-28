import Preprocessing

# input variables
comments = []
ratings = []
# Word Embedding variable, possible values: 'GloVe', 'W2V' and 'FastText'
word_embedding = 'GloVe'
# SenticNet variable, possible values: 'concept_vector', 'polarity_vector' and 'polarity_score'
sentic_net = 'concept_vector'
# model type, possible values: 0 (concept vector model), 1 (polarity vector model) and 2 (polarity score model)
model_type = 0


def main():
    Preprocessing.get_reviews(comments, ratings)
    return 0


if __name__ == "__main__":
    main()
