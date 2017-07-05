import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import linearMulti_func as linear

print('Loading data ...\n')

## Load data

data = pd.read_csv('Folds5x2_pp.csv', sep=',')
print(data)
X = data.values[:, :4]
y = data.values[:, -1].reshape(data.PE.size, 1)
m = data.PE.size
print('X:\n', X)
print('y:\n', y)
print('m:\n', m)

print('Normalizing features ....')

multi1 = linear.Function()
## Normalizing the features

##### Class file done

'''
def featureNormalize(X): # perform normalization on each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    def nor(a): # defining the normalization equation for the X
        return (a-mu)/sigma
    X_norm = np.apply_along_axis(nor, 1, X)
    return X_norm, mu, sigma
'''

##### Class file done

X_norm, mu, sigma = multi1.featureNormalize(X)
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

'''
def computeCostMulti(X, y, theta):

    m = y.size
    J = 0

    dif = (X.dot(theta)) - y

    J = 1/(2*m) * np.transpose(dif).dot(dif)
    return J
'''
### separate class file done ###

### Gradient Descent function ####
'''
def gradientDescentMulti(X, y, theta, alpha, num_iters):

    # some variables
    m = y.size
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = X.dot(theta) 

        theta[0] = theta[0] - alpha * (1/m) * (h-y).sum()
        theta[1] = theta[1] - alpha * (1/m) * ((h-y)*X[:,1]).sum()
        theta[2] = theta[2] - alpha * (1/m) * ((h-y)*X[:,2]).sum()
        theta[3] = theta[3] - alpha * (1/m) * ((h-y)*X[:,3]).sum()
        theta[4] = theta[4] - alpha * (1/m) * ((h-y)*X[:,4]).sum()

        J_history[i] = computeCostMulti(X, y, theta)
        if i%10==0:
            print(J_history[i], num_iters-i)
    return theta, J_history

'''
### separate class file done ###

theta, J_history = multi1.gradientDescentMulti(X, y, theta, alpha, num_iters)
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
