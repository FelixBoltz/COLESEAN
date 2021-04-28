import pandas as pd

reviews_file = 'comments_with_ratings.csv'


def get_reviews(comments, ratings):
    # reading csv file to access reviews
    print('Loading entire comments set:')
    reviews = pd.read_csv(reviews_file)
    comments = reviews.iloc[:, 0]
    print('Found', len(comments), 'comments')
    ratings = reviews.iloc[:, 1]
    print('Found', len(ratings), 'ratings')

