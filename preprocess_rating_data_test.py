"""
Module docstring
"""
import unittest
import pandas as pd
from preprocess_rating_data import preprocess_rating_data


class TestPreprocessRagingData(unittest.TestCase):
  """
  class docstring
  """
  def testPreprocessRatingDataWithDifferentThresholds(self):
    df = pd.DataFrame({'user_id': [1, 1, 1, 2, 2, 2, 3, 3],
                       'item_id': [0, 1, 2, 0, 1, 2, 0, 1],
                       'rating':  [3, 3, 4, 1, 3, 5, 2, 4]})

    for threshold in range(3):
      ratings = preprocess_rating_data(df, 'user_id', 'item_id', threshold)
      ratings.reset_index(drop=True, inplace=True)
      self.assertTrue(all(ratings == df))

    ratings = preprocess_rating_data(df, 'user_id', 'item_id', 3)
    ratings.reset_index(drop=True, inplace=True)
    output = pd.DataFrame({'user_id': [1, 1, 2, 2],
                           'item_id': [0, 1, 0, 1],
                           'rating':  [3, 3, 1, 3]})
    self.assertTrue(all(ratings == output))

    ratings = preprocess_rating_data(df, 'user_id', 'item_id', 4)
    ratings.reset_index(drop=True, inplace=True)
    output = pd.DataFrame({'user_id': [], 'item_id': [], 'rating': []})
    self.assertTrue(all(ratings == output))

  def testPreprocessRatingDataWithWrongUserID(self):
    # df = pd.DataFrame({'user_id': [], 'item_id': [], 'rating': []})
    pass

  def testPreprocessRatingDataWithWrongDataFrom(self):
    # df = {'user_id': [], 'item_id': [], 'rating': []}
    pass


if __name__ == '__main__':
  unittest.main()
