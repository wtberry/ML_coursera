import numpy as np

'''
Compute cost function for each iteration
'''

def costMulti(X, y, theta):

    m = y.size # some necessary values
    J = 0

    dif = (X.dot(theta)) - y 

    J = 1/(2*m) * np.transpose(dif).dot(dif) # vectorized form for cost function
    return J
