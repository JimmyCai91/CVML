"""
module docstring
"""
import numpy as np


def predict_user_ratings(user_item_matrix, user_user_matrix):
  """
  :type user_item_matrix: np.array
  :type user_user_matrix: np.array
  :rtype: np.array
  """
  # predict user ratings based on the similarity of the users
  user_item_ratings = np.zeros(user_item_matrix.shape)
  for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[1]):
      if user_item_matrix[i, j] != 0:
        # use weighted average of similar users' ratings to
        #   predict this user's rating
        sim_score = user_user_matrix[i]
        sim_users = np.argsort(sim_score)[::-1][1:6] # Top 5 similar users (excluding self), pylint: disable=line-too-long
        user_item_ratings[i, j] = np.sum(
            user_item_matrix[sim_users, j] * sim_score[sim_users])
        user_item_ratings[i, j] /= np.sum(sim_score[sim_users]) + 10**-6
  return user_item_ratings

def predcit_ratings_with_factorized_similarity(user_item_matrix, factors):
  """
  :type user_item_matrix: numpy.array
  :type item_factors: numpy.array
  :rtype: numpy.array
  """
  # predict ratings based on factorized similarities
  user_item_ratings = np.zeros(user_item_matrix.shape)
  for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[1]):
      if user_item_matrix[i, j] != 0:
        # user weighted average of similar items' ratings to
        #   predict the item's rating
        sim_scores = factors[j].dot(factors.T)
        sim_neighbors = np.argsort(sim_scores)[::-1][1:6] # Top 5 similar items (excluding self), pylint: disable=line-too-long
        user_item_ratings[i, j] = np.sum(
            user_item_matrix[i, sim_neighbors] * sim_scores[sim_neighbors])
        user_item_ratings[i, j] /= np.sum(sim_scores[sim_neighbors]) + 10**-6
  return user_item_ratings
