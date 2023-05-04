"""
module docstring
"""
import unittest
import numpy as np

def min_max_normalize(ratings):
  """
  :type ratings: numpy.array
  :rtype: numpy.array
  """
  # normalize the ratings using min-max normalization
  min_rating = ratings.min()
  max_rating = ratings.max()
  if min_rating == max_rating:
    return ratings
  else:
    normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
    return normalized_ratings

def z_score_normalize(ratings):
  """
  :type ratings: numpy.array
  :rtype: numpy.array
  """
  # normalize the ratings using z-score normalization
  mean_rating = ratings.mean()
  std_rating = ratings.std()
  if std_rating == 0:
    return ratings
  else:
    normalized_ratings = (ratings - mean_rating) / std_rating
    return normalized_ratings

def mean_normalize(ratings):
  """
  :type ratings: numpy.array
  :rtype: numpy.array
  """
  # normalize the ratings using mean normalization
  mean_rating = ratings.mean()
  normalized_ratings = ratings - mean_rating
  return normalized_ratings


class Test(unittest.TestCase):
  """
  class docstring
  """
  def testMeanNormalize(self):
    ratings = np.array([-1, 1])
    self.assertTrue(np.array_equal(mean_normalize(ratings), ratings))


if __name__ == '__main__':
  unittest.main()
