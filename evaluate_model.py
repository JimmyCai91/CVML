"""
module docstring
"""
from sklearn.metrics import mean_squared_error


def evaluate_model(user_item_matrix, user_item_ratings):
  """
  :type user_item_matrix: numpy.array
  :type user_item_ratings: numpy.array
  :rtype: float
  """
  # evaluate the performance of the model using mean squared error
  nonzero_indices = user_item_matrix.nonzero()
  mse = mean_squared_error(user_item_matrix[nonzero_indices],
                           user_item_ratings[nonzero_indices])
  return mse
