'''
Neural Network class, with 1 hidden layer
'''
from scipy import special as sp
import numpy as np

class Neural_Network(object):
    def __init__(self, input_layer_size, output_layer_size, hidden_layer_size, eps):
        # architectures
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.eps = eps

        # Weights initialization
        self.W1 = np.random.random((self.hidden_layer_size, self.input_layer_size+1))*2*self.eps-self.eps
        self.W2 = np.random.random((self.output_layer_size, self.hidden_layer_size+1))*2*self.eps-self.eps

    def forward(self, X):
        # forward propagation
        # first add bias term to X
        self.X = self.bias(X)
        self.m = self.X.shape[0]
        self.z2 = self.X.dot(self.W1.T) #(m x hidden layer size)
        self.a2 = self.sigmoid(self.z2)
        self.a2 = self.bias(self.a2)
        self.z3 = self.a2.dot(self.W2.T) #(m x outputlayer)
        a3 = self.sigmoid(self.z3)
        return a3


    def sigmoid(self, z):
        return sp.expit(z)
    
    def sigmoidGradient(self, z):
        self.gz = self.sigmoid(z)
        return self.gz*(1-self.gz)

    def costFunction(self, X, Y, lam):
        # Y: one hotted training labels
        # feedforward and make prediction
        self.y_hat = self.forward(X)
        self.lam = lam
        # compute cost and regularization
        self.m = X.shape[0]
        j = (1/self.m)*np.sum((-Y*np.log(self.y_hat)-(1-Y)*np.log(1-self.y_hat)))
        reg = (self.lam/(2*self.m))*(np.sum(self.W1[:, 1:]**2) + np.sum(self.W2[:, 1:]**2))
        j += reg
        return j

    def costGradient(self, X, Y):

        ##### backpropagation ##########

        ## initializing param
        theta1_grad = np.zeros((self.W1.shape))
        theta2_grad = np.zeros((self.W2.shape))

        D2 = np.zeros((self.output_layer_size, self.hidden_layer_size+1))
        D1 = np.zeros((self.hidden_layer_size+1, self.input_layer_size+1))

        y_hat = self.forward(X)
        self.X = self.bias(X)

        d3 = y_hat-Y #(6000x10)
        d2 = d3.dot(self.W2)*self.a2*(1-self.a2) 

        D2 += d3.T.dot(self.a2) #(10x51)
        D1 += d2.T.dot(self.X) #(51x785)
    
        theta1_grad[:, 1:] = (1/self.m)*D1[1:,1:] + (self.lam/self.m)*self.W1[:,1:] #(50x784)
        theta1_grad[:, 0] = (1/self.m)*D1[1:,0]  #(50x1)

        theta2_grad[:, 1:] = (1/self.m)*D2[:,1:] + (self.lam/self.m)*self.W2[:,1:] #(10x50)
        theta2_grad[:, 0] = (1/self.m)*D2[:,0]  #(10x1)

        grad = np.concatenate((theta1_grad.reshape(theta1_grad.size, order='F'), theta2_grad.reshape(theta2_grad.size, order='F')))

        return grad


    def getFlatWeights(self):
        # return flattened weights for scipy optimizer
        print("W1 get: ", self.W1[100, 100])
        return np.concatenate((self.W1.reshape(self.W1.size, order='F'), self.W2.reshape(self.W2.size, order='F')))
    
    def setWeights(self, params):
        # update the weights from 1D weight vector
        self.W1 = np.reshape(params[:self.hidden_layer_size * (self.input_layer_size + 1)], \
                                 (self.hidden_layer_size, self.input_layer_size + 1), order='F')
        self.W2 = np.reshape(params[self.hidden_layer_size * (self.input_layer_size + 1):], \
                                 (self.output_layer_size, self.hidden_layer_size + 1), order='F')

    def getLabel(self, prediction):
        return (prediction.argmax(axis=1)).reshape(-1, 1) ##MNIST

    def predict(self, X):
        return self.getLabel(self.forward(X))
    
    def bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]