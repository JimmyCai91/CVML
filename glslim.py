"""
GLSLIM for Recommender System
"""

import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error


class GLSLIM(object):
  """
  class docstring
  """
  def __init__(self, rating_file):
    if not rating_file.endswith('csv'):
      raise Exception("rating_file should be csv")
    self.ratings = pd.read_csv(rating_file)

  def preprocess(self):
    ratings = self.ratings

    user_counts = ratings['userId'].value_counts()
    item_counts = ratings['movieId'].value_counts()

    selected_users = ratings['userId'].isin(user_counts.index[user_counts >= 200])
    selected_items = ratings['movieId'].isin(item_counts.index[item_counts >= 200])
    ratings = ratings[selected_users & selected_items]

    self.selected_ratings = ratings[['userId', 'movieId', 'rating']]

  def glslim(self):
    ratings = self.selected_ratings
    user_item_matrix = csr_matrix(
        (ratings['rating'], (ratings['userId'], ratings['movieId'])))

    # compute the item-item similarity matrix
    item_item_matrix = user_item_matrix.T.dot(user_item_matrix)
    item_item_matrix = item_item_matrix.toarray()

    # compute the item-item similarity weights
    item_popularity = user_item_matrix.sum(axis=0)
    item_popularity = np.array(item_popularity)[0]
    item_item_weights = item_item_matrix / (item_popularity[:, np.newaxis] + 1e-8)

    # compute the user-item ratings
    user_item_ratings = user_item_matrix.dot(item_item_weights)

    # evaluate the performance
    mse = mean_squared_error(user_item_matrix.toarray(), user_item_ratings)
    print("MSE: %.4f" % mse)


class Test(unittest.TestCase):
  """
  class docstring
  """
  def testShouldRaiseExceptionWithNonCSVInput(self):
    rating_file = "ratings.xyz"
    with self.assertRaises(Exception) as context:
      GLSLIM(rating_file)
    self.assertTrue("rating_file should be csv" in str(context.exception))

  def testGLSLIM(self):
    rating_file = "ratings.csv"
    alg = GLSLIM(rating_file)
    alg.preprocess()
    alg.glslim()


if __name__ == '__main__':
  unittest.main()
