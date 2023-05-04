"""
module docstring
"""
import pandas as pd
from preprocess_rating_data import preprocess_rating_data
from compute_item_similarity import \
    compute_item_consine_similarity as compute_item_similarity
from factorization import factorize_similarity_matrix
from predict_ratings import \
    predcit_ratings_with_factorized_similarity as predict_ratings
from evaluate_model import evaluate_model_mae as evaluate_model


if __name__ == '__main__':
  # load the data
  ratings = pd.read_csv('ratings.csv')

  # preprocess the data
  ratings = preprocess_rating_data(ratings, 'userId', 'movieId', 25)

  # create the item-item matrix
  item_item_matrix = ratings.pivot_table(index='movieId', columns='userId',
                                         values='rating', fill_value=0)
  item_item_matrix = compute_item_similarity(item_item_matrix)

  # factorize the item-item similarity matrix
  item_factors = factorize_similarity_matrix(item_item_matrix)

  # create the user-item matrix
  user_item_matrix = ratings.pivot(index='userId', columns='movieId',
                                   values='rating').fillna(0).values

  # predict user-item ratings
  user_item_ratings = predict_ratings(user_item_matrix, item_factors)

  # evaluate the model
  mae = evaluate_model(user_item_matrix, user_item_ratings)
  print("MAE: %.4f" % mae)
