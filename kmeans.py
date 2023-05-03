"""
https://towardsdatascience.com/large-scale-k-means-clustering-with-gradient-descent-c4d6236acd7a
"""

import unittest
import numpy as np
from sklearn import datasets

def euclidean_distance(feat_i, feat_j):
  return np.sqrt(np.sum((feat_i - feat_j) ** 2))


class KMeansEM(object):
  """
  K-means alg. w. E (expectation) M (maximization)
  Req. X which is a d by N data matrix, K num. of clusters,
      and stop criterion delta
  1. init. cluster centers w1, w2, ..., wk, randomly
  2. E-step: fix wi and calculate cluster ids ni of each data point
  3. M-step: fix ni and calculate cluster centers wi
  4. repeat E- and M-step until the decrease of the distortion measure is less
      than delta
  """
  def __init__(self, n_cluster=3, maximum_iter=10**3, delta=10**-6):
    self.k = n_cluster
    self.iter_ub = maximum_iter
    self.stop_threshold = delta
    np.random.seed(0)

  def step_e(self, centers, x):
    labels, distances = [], []
    for feat_i in x:
      dists = [euclidean_distance(feat_i, feat_j) for feat_j in centers]
      distances.append(min(dists))
      labels.append(dists.index(distances[-1]))
    return labels, sum(distances)

  def step_m(self, labels, x):
    centers = []
    for cluster_i in range(self.k):
      sub_x = [x[j] for j, l in enumerate(labels) if l == cluster_i]
      sub_x = np.stack(sub_x).mean(axis=0)
      centers.append(sub_x)
    return centers

  def cluster(self, x):
    # Init. cluster centers
    indices = np.random.choice(len(x), size=self.k, replace=False)
    centers = [x[i] for i in indices]

    # EM
    obj1 = obj2 = None
    iter_counter = 0
    while obj1 is None or obj1 - obj2 > self.stop_threshold:
      obj1 = obj2
      labels, obj2 = self.step_e(centers, x)
      centers = self.step_m(labels, x)
      iter_counter += 1
      print("iter. {} loss {:.3f}".format(iter_counter, obj2))
      if iter_counter > self.iter_ub:
        break
    return labels, centers


class Test(unittest.TestCase):
  """
  class docstring
  """
  def test_euclidean_distance(self):
    feat_i = np.array([1, 1, 1])
    feat_j = np.array([2, 1, 1])
    self.assertEqual(0, euclidean_distance(feat_i, feat_i))
    self.assertEqual(1, euclidean_distance(feat_i, feat_j))
    self.assertEqual(
        euclidean_distance(feat_i, feat_j),
        euclidean_distance(feat_j, feat_i)) # pylint: disable=arguments-out-of-order

  def test_digits_classification(self):
    data, target = datasets.load_digits(return_X_y=True)
    (n_samples, _), n_digits = data.shape, np.unique(target).size
    clf = KMeansEM(n_cluster=n_digits, delta=10**-5)
    cluster_labels, cluster_centers = clf.cluster(data)

    # post processing
    y_pred = [0] * len(target)
    for i in range(n_digits):
      min_dist, min_label = float("inf"), -1
      for label, j in zip(target, data):
        dist = euclidean_distance(cluster_centers[i], j)
        if dist < min_dist:
          min_dist = dist
          min_label = label
      for j, l in enumerate(cluster_labels):
        if l == i:
          y_pred[j] = min_label

    acc = np.sum(target == y_pred) / n_samples
    print("K-Means classification accuracy", acc)


if __name__ == '__main__':
  unittest.main()


