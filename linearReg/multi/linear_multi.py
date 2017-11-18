import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import linearMulti_func as lin

print('Loading data ...\n')

## Load data

data = pd.read_csv('CCPP_dataset/Folds5x2_pp.csv', sep=',')
print(data)
X = data.values[:, :4]
y = data.values[:, -1].reshape(data.PE.size, 1)
m = data.PE.size
print('X:\n', X)
print('y:\n', y)
print('m:\n', m)

print('Normalizing features ....')

## Normalizing the features

X_norm, mu, sigma = lin.featureNormalize(X)
print('X_norm\n', X_norm)
print('mu\n', mu)
print('sigma\n', sigma)

# Add intercept term to X
X = np.c_[np.ones((m, 1)), X_norm]
print('X:\n', X)

############# gradient descent ################
print('Running gradient descent ....')

# choose some alpha value

alpha = 1e-2
alpha2 = 9e-1
alpha3 = 1e-3
num_iters = 400

# theta values

theta = np.zeros((X[0].size, 1))
theta2 = np.zeros((X[0].size, 1))
theta3 = np.zeros((X[0].size, 1))

print('theta2\n', theta2)
### Cost Function function ###


### Gradient Descent function ####

theta, J_history = lin.gradientDescentMulti(X, y, theta, alpha, num_iters)
#theta2, J_history2 = gradientDescentMulti(X, y, theta2, alpha2, num_iters)
#theta3, J_history3 = gradientDescentMulti(X, y, theta3, alpha3, num_iters)
print(J_history)

### Plot the cost function graph ###

plt.figure()
plt.plot(np.arange(1, J_history.size+1), J_history, 'b')
#plt.plot(np.arange(1, J_history2.size+1), J_history2, 'c')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.grid(True)
plt.show()

### Display gradient descent's result ###

print('theta computed from gradient descent:')
print(theta)
#print('theta2 computed from gradient descent:')
#print(theta2)
#print('theta3 computed from gradient descent:')
#print(theta3)

sample = np.array([1, 8.34, 40.77, 1010.84, 90.01]).reshape(1, 5)
price = sample.dot(theta)
print('predicted price by g-descent is', price)
