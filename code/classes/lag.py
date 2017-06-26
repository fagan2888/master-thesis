import numpy as np

def lag(a, p=1):
  a = np.asanyarray(a)

  # n: number of observations
  # m: number of variables that need to be lagged
  n, m = a.shape

  ret_a = np.zeros( (n - p + 1, p*m) )

  for i in range(p-1, n):
    ret_a[i - p + 1] = np.concatenate([ [a[i - k][j] for k in range(0, p)] for j in range(0, m)])


  return ret_a




# data = [[1, 5, 10], [2, 6, 11], [3, 7, 12], [4, 8, 13], [5, 9, 14]]

# print(lag(data, p=1))

# data = [[1], [2], [3], [4], [5], [7], [20], [25]]

# print(lag(data, p=2))
