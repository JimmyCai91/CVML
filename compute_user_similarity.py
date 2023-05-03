"""
module docstring
"""
import numpy as np
from scipy.spatial.distance import cosine

def compute_user_similarity(user_item_matrix):
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
