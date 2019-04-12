'''
Neural Network from scratch in python, numpy, pandas, and scipy
'''
### Neural Network exercise using fashion MNIST dataset ###

# based on Coursera's Machine Learning course, exercise 4, Neural Network
# 

#### NOTE ###
'''
This script runs optimization algorithm on FULL dataset, not like batch / stochastic 
gradient descent. As a result it'll take sinificant amout of time to finish training. 
It took 30 minutes 30 seconds to train 100 iteration, on a Laptop with dual core 7th gen i7. 
'''
import scipy.io
import pandas as pd # library to manage data
from scipy.optimize import minimize # optimizer for training NN
from scipy import special as sp
import numpy as np # matrices calculation
import matplotlib.pyplot as plt # graphing

### now define functions needed in the future


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
    performs Feedforward, backprop and return the cost in the variable J.
    also stores cost of each iteration in j_hist list
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

    # prediction
    p = (a3.argmax(axis=1)).reshape(m, 1)
    print('prediction: ', p)

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

    # records cost and accuracy as history
    j_hist.append(J)
    test_acc.append((p==y).astype(float).mean()*100)


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


def visualize(rows, columns, data):
    amount = rows*columns
    image = np.zeros((amount, 28, 28))
    
    for i in range(amount):
        image[i] = data[i, :].reshape(-1, 28)
    
    fig = plt.figure()

    for i in range(amount):
        ax = fig.add_subplot(rows, columns, 1+i)
        plt.imshow(image[i], cmap='viridis')
        #plt.matsho(image[i], cmap='viridis')
        plt.axis('off')
        plt.sca(ax)

    plt.show()


import fashion_data_import as fin # data importing script
import time # to record time took to train

### first, import fashion MNIST dataset

##### Training Sets ##################################################

# loading the MNIST data, converting them into numpy arrays


### Loading the training dataset
X, y = fin.data_in('train')
m, n = X.shape #(6000, 784), m for number of training examples, n for number of features
print('X: \n', X)
print('y: \n', y)

### Loading the testing dataset
Xt, yt = fin.data_in('test')
mt, nt = Xt.shape
Xt = np.c_[np.ones((mt, 1)), Xt] ## adding bias term to Xt

### setting up parameters for Neural Network ###
# 3 Layers total, input, 1 hidden and output layer.
input_layer_size = 784 # 28x28 input images of digits
hidden_layer_size = 784 # hidden units
num_labels = 10 # classify into 10 outputs, same as number of output layer's units

## visualize the dataset
visualize(10, 10, Xt[:100, 1:])

### setting cloths name labels for # labels for human interpretation
human_label = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'} 
    ## here, I'm associating each neural network output label # to name of each outputs.

# display randomly selected datapoints as a picture
'''
num_of_image = 0
for pic in range(num_of_image): # num_of_image amount of images displayed
    sample = np.random.choice(X.shape[0]) # choosing random index num for X
    pixel = X[sample, :].reshape(-1, int(np.sqrt(n))) # reshaping the one sample into (28x28)
    y_val = y[sample, 0] # y label value
    lab = 'label: ' + str(y_val)
    plt.imshow(pixel, cmap='Greys')
    ## imshow colormap https://matplotlib.org/examples/color/colormaps_reference.html

    plt.xlabel(lab + ' ' +  str(human_label[y_val]))
    plt.axis('on')
    plt.show()
'''

# Adding bias terms to X, training dataset 
X = np.c_[np.ones((m, 1)), X] ## adding bias term to X
print('X with bias: \n', X)
'''
##### Initializing Neural Network Parameters ########################

# epsilon for initial theta
eps = 0.5
print('initializing Neural Network Parameters.....')
i_theta1 = np.random.random((hidden_layer_size, n+1))*2*eps-eps
i_theta2 = np.random.random((num_labels, hidden_layer_size+1))*2*eps-eps
print('initial theta1: \n', i_theta1)

# here, theta matrices are folded into one long vector?? to feed into optimization algorithm
nn_params = np.concatenate((i_theta1.reshape(i_theta1.size, order='F'), i_theta2.    reshape(i_theta2.size, order='F')))
##### Compute Cost (Feedforward) #########################
print('Feedforward Using Neural Network ....')

# weight regularization parameter 
# adjust this to prevent overfitting. 
lam = 0

print('making sure the algorithm implementation is correct...')
print()

########## algorithm checking with coursera dataset #########

# trial feedforward
cost, grad = fun.CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
print('lambda with 0')
print('cost at parameters (from ex4weights), should be about 0.287629')
print('cost: \n', cost)

lam = 1
print('checking cost function with regularization...')
cost1, grad = fun.CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
print('lambda with 1')
print('cost at parameters (from ex4weights), should be about 0.383770')
print('cost: \n', cost1)

##### sigmoid gradient #######################
print('Evaluating sigmoid gradient....\n')

g = fun.sigmoidGradient(np.array((-1, -0.5, 0, 0.5, 1)))
print('Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]: \n', g)

## imshow colormap https://matplotlib.org/examples/color/colormaps_reference.html

##### Implementing regularization ##############
# assuming that the BackPropagation is right...
print('checking BackPropagation with regularization...')
lam = 3
debug_J, grad = fun.CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

print('cost at debugging parameters with lambda: ', lam)
print('Cost: ', debug_J)
print('This value should be about: 0.576051')
'''

''' Code begin from here again!!!'''

##### Initializing Parameters #################
eps = 0.5
print('initializing Neural Network Parameters.....')

## initializing random values of theta here
r_theta1 = np.random.random((hidden_layer_size, n+1))*2*eps-eps
r_theta2 = np.random.random((num_labels, hidden_layer_size+1))*2*eps-eps
print('initial theta1: \n', r_theta1)

## "flattening" the theta matrices, so it can be fed to optimization algorithm
r_nn_params = np.concatenate((r_theta1.reshape(r_theta1.size, order='F'), r_theta2.reshape(r_theta2.size, order='F')))

##### Training NN ###########################


### Use python optimizing function


mymethod = 'L-BFGS-B' # specifying the optimization algorithm
maxiter = 20
lam =0.9 
j_hist = [] # recording history of cost over the training
test_acc = []
print('training NN by ', mymethod, '.....')
print('Starting training timer....')

start_time = time.time() # setting stopwatch for training
## arguments for the optimization algorithm
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lam) # args for miminize
results = minimize(CostFunction, x0=r_nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method=mymethod, jac=True)
nn_params = results['x']


####### Making Prediction based on learned theta values #################

# flattening the theta matrics
theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                         (hidden_layer_size, input_layer_size + 1), order='F')
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                         (num_labels, hidden_layer_size + 1), order='F')

### Training set Prediction ###
prediction = predict(theta1, theta2, X)
accuracy = (prediction == y).astype(float).mean()*100

print('Training set accuracy: \n', accuracy, '%')

### Testing set Prediction ###
prediction_test = predict(theta1, theta2, Xt)
print("done prediction, now calculate acc...")
accuracy_test = (prediction_test == yt).astype(float).mean()*100

### Printing out parameters ###
elapsed_time = time.time() - start_time
minutes = elapsed_time//60
seconds = minutes%60
print("Time elapsed during training: ", minutes, "minutes", seconds, 's')
print('Testing set accuracy: \n', accuracy_test, '%')
print('Machine Learning Parameters...')
print('Sizes of the NN: input:', input_layer_size, ' hidden:', hidden_layer_size, ' output:', num_labels)
print('Lambda: ', lam)
print('# of iteration: ', maxiter)
print('Optimization Algorithm: ', mymethod)

##### plotting cost function graph ########
plt.figure()
plt.plot(np.arange(1, len(j_hist)+1), j_hist, 'b', label='cost')
plt.plot(np.arange(1, len(test_acc)+1), test_acc, 'r', label='accuracy')
plt.legend()
plt.xlabel('# of iterations')
plt.ylabel('Cost & Accuracy')
plt.grid(True)
plt.show()


'''
for pic in range(mt): # keep printing out the graph, and prediction until user press ctrl-C
    sample = np.random.choice(Xt.shape[0]) # choosing random index num for X
    pixel = Xt[sample, 1:].reshape(-1, int(np.sqrt(n))) # reshaping the one sample into (28x28)
    y_val = yt[sample, 0] # y label value
    pt_val = prediction_test[sample, 0] # testing dataset prediction label value
    lab_answer = 'correct label: ' + str(y_val)
    lab_prediction = 'predicted label: ' + str(pt_val)
    plt.imshow(pixel, cmap='Greys')

    plt.xlabel(lab_answer + ' ' +  str(human_label[y_val]))
    plt.ylabel(lab_prediction + ' ' +  str(human_label[pt_val]))
    plt.axis('on')
    print('predicted label: ', prediction_test[sample, 0])
    print('correct label: ', y_val)
    plt.show()
'''