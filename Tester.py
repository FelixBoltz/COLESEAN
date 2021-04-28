import Preprocessing
from sklearn.model_selection import train_test_split

# Word Embedding variable, possible values: 'GloVe', 'W2V' and 'FastText'
word_embedding = 'GloVe'
# SenticNet variable, possible values: 'concept_vector', 'polarity_vector' and 'polarity_score'
sentic_net = 'concept_vector'
# model type, possible values: 0 (concept vector model), 1 (polarity vector model) and 2 (polarity score model)
model_type = 0


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
    return 0


if __name__ == "__main__":
    main()
