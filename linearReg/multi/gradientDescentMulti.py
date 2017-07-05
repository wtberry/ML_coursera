import numpy as np

'''
performs gradient descent on given X, y, theta, alpha for num_liter times
'''
def descentMulti(X, y, theta, alpha, num_liters):

    # some variables
    m = y.size
    J_history = np.zeros((num_liters, 1))

    for i in range(num_iters):
        predictions = X.dot(theta)

        theta[0] = theta[0] - alpha * (1/m) * (predictions - y).sum()
        theta[1] = theta[1] - alpha * (1/m) * ((predictions - y)*X[:,1]).sum()
        theta[2] = theta[2] - alpha * (1/m) * ((predictions - y)*X[:,2]).sum()
        theta[3] = theta[3] - alpha * (1/m) * ((predictions - y)*X[:,3]).sum()
        theta[4] = theta[4] - alpha * (1/m) * ((predictions - y)*X[:,4]).sum()

        J_history[i] = computeCostMulti(X, y, theta)
        if i% == 0:
            print(J_history[i], num_iters-i)

    return theta, J_history
