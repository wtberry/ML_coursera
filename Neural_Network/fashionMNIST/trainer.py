'''
class for training setup and training
'''
from scipy import optimize
import numpy as np

class trainer(object):
    def __init__(self, N, X, y):
        # set up model
        self.N = N
        self.J = [] # empty list for cost
        self.acc = [] # training accuracy

        # set up data
        self.X = X
        self.y = y

    def callback(self, params):
        # update weights and accumulate cost record
        self.N.setWeights(params)
        self.J.append(self.N.costFunction(self.X, self.y, self.lam))
        self.acc.append(self.accuracy(self.N.predict(self.X), self.label))

    def costFunctionWrapper(self, params, X, y, lam):
        # function to pass into scipy optimizer.
        # compute cost, gradients and return it
        self.N.setWeights(params)
        cost = self.N.costFunction(X, y, lam)
        grad = self.N.costGradient(X, y)
        return [cost, grad]

    def one_hot(self, y):
        Y = np.zeros((y.shape[0], np.unique(y).size))
        for i in range(y.size):
            Y[i, y[i]]=1   # MNIST datasetk
        return Y

    def accuracy(self, p, y):
        comp = p == y
        comp_o_zero = comp.astype(float)
        accuracy = comp_o_zero.mean()*100
        return accuracy

    def train(self, lam, maxiter, mymethod, label_one_hot=True):
        '''
        train the model with provided data and parameters
        Params:
        - lam: lamda, the regularization: float
        - maxiter: maximum iteration number for optimization: int
        - mymethod: optimization method name: string
        - label_one_hot: weather one hot y label or not: bool
        '''


        if label_one_hot:
            self.label = self.y
            self.y = self.one_hot(self.y)
        self.lam = lam
        
        params0 = self.N.getFlatWeights()
        myargs = (self.X, self.y, self.lam)

        results = optimize.minimize(self.costFunctionWrapper, x0=params0, \
            args=myargs, callback=self.callback, options={'disp': True, 'maxiter':maxiter}, method=mymethod, jac=True)

        self.N.setWeights(results.x)
        self.optimizationResults = results