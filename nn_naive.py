import numpy as np
import time

def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input data
X = np.array([[0, 0, 1, 2],
              [0, 1, 1, 4],
              [1, 0, 1, 9],
              [1, 1, 1, -4]])

# output data
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])

y = np.sin(np.sum(np.power(X, 3), axis=1)).reshape((4, 1))
print('y is', y, y.shape)

np.random.seed(1)

#synapses
syn0 = 2*np.random.random((4, 13)) - 1
syn1 = 2*np.random.random((13, 1)) - 1

#training step
for j in range(60000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2

    if (j%10000)==0:
        print('Error', np.mean(np.abs(l2_error)))

    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print('Output', l2)

start_t = time.time()
for i in range(10000):
    outcomes = [3, 5, 7, 9, 11, 99, 88, 128, 7]
    outcomes.remove(9)
end_t = time.time()
print('cost time:', end_t-start_t)
