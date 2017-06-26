import numpy as np
import math
import random

def gen_batches_idx(data, n_buckets):
  a = np.asanyarray(data)
  idxs = list(range(0, a.shape[0]))
  np.random.shuffle(idxs)
  return np.array_split(idxs, n_buckets)
  # np.random.shuffle(a)
  # return np.array_split(a, size)

def get_batch(datas, batch_idxs):
  # a = np.asanyarray(data)
  # return a[batch_idxs]
  # print(datas[0].shape)
  # print(datas[1].shape)
  return [np.asanyarray(data)[batch_idxs] for data in datas]




# data1 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
# data2 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
# data = [data1, data2]

# batches = gen_batches_idx(np.matrix(data1), 3)
# print(get_batch(data, batches[0]))
# print(get_batch(data, batches[1]))
# print(get_batch(data, batches[2]))


