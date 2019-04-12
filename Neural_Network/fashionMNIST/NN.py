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
from scipy.optimize import minimize # optimizer for training NN
import pandas as pd # library to manage data
import numpy as np # matrices calculation
import matplotlib.pyplot as plt # graphing
import time # to record time took to train

import functions as fun # bunch of functions needed for NN
import ac # calculates accuracy  of the model after training
import fashion_data_import as fin # data importing script

from NeuralNetwork import Neural_Network
from trainer import trainer

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
#Xt = np.c_[np.ones((mt, 1)), Xt] ## adding bias term to Xt

### setting up parameters for Neural Network ###
# 3 Layers total, input, 1 hidden and output layer.
input_layer_size = 784 # 28x28 input images of digits
hidden_layer_size = 500 # hidden units
output_layer_size = 10 # classify into 10 outputs, same as number of output layer's units

### setting cloths name labels for # labels for human interpretation
human_label = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'} 
    ## here, I'm associating each neural network output label # to name of each outputs.

# display randomly selected datapoints as a picture
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

# Adding bias terms to X, training dataset 
#X = np.c_[np.ones((m, 1)), X] ## adding bias term to X
#print('X with bias: \n', X)
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
#eps = 0.5
print('initializing Neural Network .....')
Net = Neural_Network(input_layer_size, output_layer_size, hidden_layer_size)

## initializing random values of theta here
#r_theta1 = np.random.random((hidden_layer_size, n+1))*2*eps-eps
#r_theta2 = np.random.random((num_labels, hidden_layer_size+1))*2*eps-eps
print('initial theta1: \n', Net.W1)

## "flattening" the theta matrices, so it can be fed to optimization algorithm
#r_nn_params = np.concatenate((r_theta1.reshape(r_theta1.size, order='F'), r_theta2.reshape(r_theta2.size, order='F')))

##### Training NN ###########################


### Use python optimizing function
print("initializing trainer....")
mymethod = 'L-BFGS-B' # specifying the optimization algorithm
maxiter = 10
lam =0.9 
print('training NN by ', mymethod, '.....')
t = trainer(Net)

print('Starting training timer....')
start_time = time.time() # setting stopwatch for training
t.train(X, y, lam, maxiter, mymethod, label_one_hot=True)
## arguments for the optimization algorithm
### Printing out parameters ###
elapsed_time = time.time() - start_time
'''
nn_params = results['x']


####### Making Prediction based on learned theta values #################

# flattening the theta matrics
theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                         (hidden_layer_size, input_layer_size + 1), order='F')
theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                         (num_labels, hidden_layer_size + 1), order='F')

'''

### Training set Prediction ###
accuracy = ac.accuracy(Net.predict(X), y)

print('Training set accuracy: \n', accuracy, '%')

### Testing set Prediction ###
accuracy_test = ac.accuracy(Net.predict(Xt), yt)

minutes = elapsed_time//60
seconds = minutes%60
print("Time elapsed during training: ", minutes, "minutes", seconds, 's')
print('Testing set accuracy: \n', accuracy_test, '%')
print('Machine Learning Parameters...')
print('Sizes of the NN: input:', input_layer_size, ' hidden:', hidden_layer_size, ' output:', output_layer_size)
print('Lambda: ', lam)
print('# of iteration: ', maxiter)
print('Optimization Algorithm: ', mymethod)

'''
NOt working, so commented out 
##### plotting cost function graph ########
plt.figure()
plt.plot(np.arange(1, J_hist.size+1), J_hist, 'b')
plt.xlabel('# of iterations')
plt.ylabel('Cost J')
plt.grid(True)
plt.show()
'''


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