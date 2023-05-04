"""
module docstring
"""

import unittest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def factorize_similarity_matrix(similarity_matrix, num_factors=50):
  """
  :type similarity_matrix: numpy.array
  :type num_factors: int
  :rtype: numpy.array
  """
  # factorize the similarity matrix using SVD
  svd = TruncatedSVD(n_components=num_factors)
  factors = svd.fit_transform(similarity_matrix)
  factors = normalize(factors, axis=1)
  return factors


class Test(unittest.TestCase):
  """
  class docstring
  """
  def testFactorizeSimilarityMatrix(self):
    pass


if __name__ == "__main__":
  unittest.main()
