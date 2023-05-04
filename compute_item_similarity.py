"""
module docstring
"""
import unittest
from sklearn.metrics.pairwise import cosine_similarity


def compute_item_consine_similarity(item_item_matrix):
  """
  :type item_item_matrix: numpy.array
  :rtype: numpy.array
  """
  # create the item-item similarity matrix with Z-score normalization
  item_item_matrix = item_item_matrix.apply(
      lambda x: (x - x.mean()) / (x.std() + 10**-6), axis=1)
  item_item_matrix = cosine_similarity(item_item_matrix)
  return item_item_matrix


class Test(unittest.TestCase):
  def testItemCosineSimilarity(self):
    pass


if __name__ == '__main__':
  unittest.main()
