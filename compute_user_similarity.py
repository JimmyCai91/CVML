"""
module docstring
"""
import unittest
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances

def compute_user_cosine_similarity(user_item_matrix):
  """
  : type user_item_matrix: numpy.array`
  : rtype: numpy.array
  """
  # compute the user-user similarity matrix
  #     using cosine similarity
  user_user_matrix = np.zeros((user_item_matrix.shape[0],
                               user_item_matrix.shape[0]))
  for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
      user_user_matrix[i, j] = 1 - cosine(user_item_matrix[i],
                                          user_item_matrix[j])
  return user_user_matrix

def compute_user_pearson_correlation_similarity(user_item_matrix):
  """
  : type user_item_matrix: numpy.array`
  : rtype: numpy.array
  """
  # compute the user-user similarity matrix
  #     using Pearson correlation
  user_user_matrix = 1 - pairwise_distances(user_item_matrix,
                                            metric='correlation')
  return user_user_matrix

def compute_user_mean_squared_similarity(user_item_matrix):
  # Compute the user-user similarity matrix using mean squared difference
  user_user_matrix = np.zeros((user_item_matrix.shape[0],
                               user_item_matrix.shape[0]))
  for i in range(user_item_matrix.shape[0]):
    for j in range(user_item_matrix.shape[0]):
      numerator, denominator = 0, 0
      for ru, rv in zip(user_item_matrix[i], user_item_matrix[j]):
        if ru != 0 and rv != 0:
          numerator += 1
          denominator += (ru - rv)**2
      user_user_matrix[i, j] = numerator / (denominator + 10**-6)
  return user_user_matrix


class TestComputeUserSimilarity(unittest.TestCase):
  """
  class docstring
  """
  def testComputeUserCosineSimilarity(self):
    user_item_matrix = np.array([[0, 1], [1, 0]])
    user_user_matrix = compute_user_cosine_similarity(user_item_matrix)
    output = np.array([[1, 0], [0, 1]])
    self.assertTrue((user_user_matrix == output).all())

    user_item_matrix = np.array([[0, 1], [0, 1]])
    user_user_matrix = compute_user_cosine_similarity(user_item_matrix)
    output = np.array([[1, 1], [1, 1]])
    self.assertTrue((user_user_matrix == output).all())

  def testComputeUserPearsonCorrelation(self):
      pass


if __name__ == '__main__':
  unittest.main()
