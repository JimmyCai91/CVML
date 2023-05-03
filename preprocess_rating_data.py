"""
module docstring
"""

def preprocess_rating_data(ratings, user_id, item_id, threshold=5):
  """
  preprocess the data by removing users and items with too few
      ratings
  """
  user_counts = ratings.groupby(user_id).size()
  item_counts = ratings.groupby(item_id).size()
  ratings = ratings[
      (ratings[user_id].isin(user_counts.index[user_counts >= threshold])) &
      (ratings[item_id].isin(item_counts.index[item_counts >= threshold]))]
  return ratings
