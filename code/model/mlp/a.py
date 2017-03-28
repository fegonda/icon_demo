import numpy as np
import partition_comparison
from math import log

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma), abs(sigma) / log(n, 2)


if __name__ == '__main__':
    print 'creating baseline performance project test'

    X1 = [ [1,2,3,4,5], [1,7,8,9,10] ]
    Y1 = [ [1,2,3,4,5], [1,7,8,9,10] ]

    X1 = np.ones(10)
    Y1 = np.ones(10)

    print X1
    print Y1

    X1 = [X1]
    Y1 = [Y1]
    print(variation_of_information(X1, Y1))

    X1 = np.ones(10)
    Y1 = np.ones(10)

    print X1
    print Y1

    X1 = np.array([1,2,3,4,5,1])
    Y1 = np.array([1,3,2,2,5,0])
    print(partition_comparison.variation_of_information(X1, Y1))
