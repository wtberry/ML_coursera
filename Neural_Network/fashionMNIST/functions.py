import numpy as np
from scipy import special as sp

'''
bunch of functions needed for Neural Network
'''

## sigmoid funcgtion
def sigmoid(z):
    '''
    returns sigmoid function value, by using scipy.special.expit
    It takes care of runtime overflow error
    '''
    return sp.expit(z)

def sigmoidGradient(z):
    gz = sigmoid(z)
    return gz*(1-gz)

def CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
    '''
    performs Feedforward and return the cost in the variable J.
    '''
    ### reshape nn_params back into the theta1 and 2

    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                                 (hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                                 (num_labels, hidden_layer_size + 1), order='F')

    m = X.shape[0]
    ### initial values, to be returned correctly
    J = 0

    theta1_grad = np.zeros((theta1.shape))
    theta2_grad = np.zeros((theta2.shape))

    ## feedforwarding, input to hidden
    z2 = X.dot(theta1.T) #(60000x50) 
    a2 = sigmoid(z2) # same shape
    a2 = np.c_[np.ones((m, 1)), a2] # adding bias term

    ## feedforward, hidden to output
    z3 = a2.dot(theta2.T) #(60000x10)
    a3 = sigmoid(z3)

    ## converting y label vector into 1/0 matrix of (60000x10), 1 at the index of the num

    Y = np.zeros(a3.shape)
    for i in range(y.size):
        Y[i, y[i]]=1   # MNIST dataset
    

    ## calculating the cost, without regularization, vectorized :)
    j = (1/m)*np.sum((-Y*np.log(a3)-(1-Y)*np.log(1-a3)))
    
    ## regularization
    regularization = (lam/(2*m))*(np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))

    ## cost with regularization
    J = j + regularization

    ##### backpropagation ##########

    ## initializing param
    D2 = np.zeros((num_labels, hidden_layer_size+1))
    D1 = np.zeros((hidden_layer_size+1, input_layer_size+1))
    '''
    for i in range(m):
        d3 = (a3[i, :] - Y[i, :]).reshape(1, num_labels) # (1x10)
        d2 = (theta2.T.dot(d3.T)*a2[i, :].reshape(hidden_layer_size+1, 1)*(1-a2[i,:].reshape(hidden_layer_size+1, 1)))  #(51x1)

        D2 = D2+d3.T.dot(a2[i, :].reshape(1, hidden_layer_size+1)) #(10x51) + (10x1).dot(1x51)
        D1 = D1+d2.dot(X[i, :].reshape(1, input_layer_size+1))
    '''

    d3 = a3-Y #(6000x10)
    d2 = d3.dot(theta2)*a2*(1-a2) 

    D2 += d3.T.dot(a2) #(10x51)
    D1 += d2.T.dot(X) #(51x785)
    
    theta1_grad[:, 1:] = (1/m)*D1[1:,1:] + (lam/m)*theta1[:,1:] #(50x784)
    theta1_grad[:, 0] = (1/m)*D1[1:,0]  #(50x1)

    theta2_grad[:, 1:] = (1/m)*D2[:,1:] + (lam/m)*theta2[:,1:] #(10x50)
    theta2_grad[:, 0] = (1/m)*D2[:,0]  #(10x1)

    grad = np.concatenate((theta1_grad.reshape(theta1_grad.size, order='F'), theta2_grad.reshape(theta2_grad.size, order='F')))

    return [J, grad]

def predict(theta1, theta2, X):
    m = X.shape[0]
    num_labels = theta2.shape[0]

    p = np.zeros((m, 1))

    ## feedforwarding, input to hidden
    z2 = X.dot(theta1.T) #(60000x50) 
    a2 = sigmoid(z2) # same shape
    a2 = np.c_[np.ones((m, 1)), a2] # adding bias term

    ## feedforward, hidden to output
    z3 = a2.dot(theta2.T) #(60000x10)
    a3 = sigmoid(z3)

    p = (a3.argmax(axis=1)).reshape(m, 1) ##MNIST
    return p

def costHist(J, hist):
    hist.append(J)
