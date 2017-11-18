import numpy as np

### sigmoid 

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

### prediction

def predict(theta1, theta2, X):
    '''
    Predict predict the label of an input given a trained neural network
    The function outputs the predicted label of X given the trained weights of
    a neural network
    '''

    # useful values
    m = X.shape[0] # == 5000

    # return the following value correctly
    p = np.zeros((m, 1)) 

    # adding ones to the column
    X = np.c_[np.ones((m, 1)), X] # (5000*401)

    a2 = sigmoid(X.dot(theta1.T)) # (5000, 25)

    # adding intercept term 
    a2 = np.c_[np.ones((m, 1)), a2] #(5000, 26)

    # computing the a3
    a3 = sigmoid(a2.dot(theta2.T)) #(5000, 10)

    p = (a3.argmax(axis=1)+1).reshape(m, 1) # getting max values and adding 1 for python and octave index difference   
    return p 
