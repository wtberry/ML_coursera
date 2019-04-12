'''
class for training setup and training
'''
from scipy import optimize
import numpy as np

class trainer(object):
    def __init__(self, N):
        # set up model
        self.N = N
        self.J = [] # empty list for cost

    def callback(self, params):
        # update weights and accumulate cost record
        self.N.setWeights(params)
        self.J.append(self.N.costFunction(self.X, self.y, self.lam))

    def costFunctionWrapper(self, params, X, y, lam):
        # function to pass into scipy optimizer.
        # compute cost, gradients and return it
        self.N.setWeights(params)
        cost = self.N.costFunction(X, y, lam)
        grad = self.N.costGradient(X, y)
        self.J.append(cost)
        return [cost, grad]

    def one_hot(self, y):
        Y = np.zeros((y.shape[0], np.unique(y).size))
        for i in range(y.size):
            Y[i, y[i]]=1   # MNIST datasetk
        return Y

    def train(self, X, y, lam, maxiter, mymethod, label_one_hot=True):
        '''
        train the model with provided data and parameters
        Params:
        - lam: lamda, the regularization: float
        - maxiter: maximum iteration number for optimization: int
        - mymethod: optimization method name: string
        - label_one_hot: weather one hot y label or not: bool
        '''

        # set up data
        self.X = np.c_[np.ones((y.shape[0], 1)), X]

        if label_one_hot:
            self.y = self.one_hot(y)
        else:
            self.y = y
        self.lam = lam
        
        params0 = self.N.getFlatWeights()
        myargs = (self.X, self.y, self.lam)

        results = optimize.minimize(self.costFunctionWrapper, x0=params0, \
            args=myargs, options={'disp': True, 'maxiter':maxiter}, method=mymethod, jac=True)

        self.N.setWeights(results.x)
        self.optimizationResults = results