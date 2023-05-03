"""
module string
"""
import unittest
import numpy as np
from compute_user_similarity import compute_user_similarity

class TestComputeUserSimilarity(unittest.TestCase):
  """
  class docstring
  """
  def testComputeUserSimilarity(self):
    user_item_matrix = np.array([[0, 1], [1, 0]])
    user_user_matrix = compute_user_similarity(user_item_matrix)
    output = np.array([[1, 0], [0, 1]])
    self.assertTrue((user_user_matrix == output).all())

    user_item_matrix = np.array([[0, 1], [0, 1]])
    user_user_matrix = compute_user_similarity(user_item_matrix)
    output = np.array([[1, 1], [1, 1]])
    self.assertTrue((user_user_matrix == output).all())


if __name__ == '__main__':
  unittest.main()
