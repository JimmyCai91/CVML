"""
module docstring
"""
import pandas as pd
from preprocess_rating_data import preprocess_rating_data
from rating_normalizations import min_max_normalize as normalize
from compute_user_similarity import \
    compute_user_cosine_similarity as compute_user_similarity
from predict_ratings import \
    predict_user_ratings as predict_ratings
from evaluate_model import evaluate_model_mae as evaluate_model


if __name__ == '__main__':
  # load the data
  ratings = pd.read_csv('ratings.csv')

  # preprocess the data
  ratings = preprocess_rating_data(ratings, 'userId', 'movieId', 25)

  # create the user-item matrix
  user_item_matrix = ratings.pivot(index='userId', columns='movieId',
                                   values='rating').fillna(0).values

  # compute the user-user similarity matrix
  for i in range(user_item_matrix.shape[0]):
    user_item_matrix[i] = normalize(user_item_matrix[i])
  user_user_matrix = compute_user_similarity(user_item_matrix)

  # predict user-item ratings
  user_item_ratings = predict_ratings(user_item_matrix, user_user_matrix)

  # evaluate the model
  mae = evaluate_model(user_item_matrix, user_item_ratings)
  print("MAE: %.4f" % mae)
