# ex3, neural network 
# complete this and predict.py

import scipy.io
import numpy as np
import predict_func as p
import matplotlib.pyplot as plt

### setting up useful parameters 

input_layer_size = 400 # 20*20 input images of digits
hidden_layer_size = 25
num_labels = 10

# Part 1, loading the data

print('loading the data.....\n')

data = scipy.io.loadmat('ex3data1.mat')
X = data['X'] # numpy array
y = data['y'] # numpy array
n = X.shape[1]
print('X:\n', X, '\n', X.shape)
print('y:\n', y, '\n', y.shape)


### Part 2, loading paramerters and showing sample handwritings

# We load some pre-installed neural network parameters

print('loading saved Neural Network parameters')

# load the weights into variables Theta1 and Theta2

w = scipy.io.loadmat('ex3weights.mat')
theta1 = w['Theta1']
theta2 = w['Theta2']
print('theta1:\n', theta1, '\n', theta1.shape)
print('theta2:\n', theta2, '\n', theta2.shape)

for pic in range(3):
    sample = np.random.choice(X.shape[0])
    pixel = X[sample, :].reshape(-1, 20)
    plt.imshow(pixel.T)
    plt.axis('off')
    plt.show()

### Part 3: Implement Predict
'''
using 'predict' function to use neural network to predict the labels of the training set. This lets you compute the trainnign set accuracy.
'''


pred = p.predict(theta1, theta2, X)
print('a3', pred)

comp = pred == y # comparing pred with y value, return boolean
comp_binary = comp.astype(float) # convert boolean val to float, 1 or 0
accuracy = comp_binary.mean() * 100 # calculate mean value and %

print('Training Set Accuracy: \n', accuracy, '%')

'''
run through examples of training set to see which one it's predicrting, 
and plot the picture
'''

m = X.shape[0]
sample2 = np.random.choice(m, size=m)
for e in sample2:
    pr = p.predict(theta1, theta2, X[e, :].reshape(1, n))
    pix = X[e, :].reshape(-1, 20)
    plt.imshow(pix.T)
    plt.axis('off')
    print('\nNN\'s prediction:', pr[0, 0])
    print('compare y and prediction:', pr[0, 0]==y[e])
    print('press ctrl+c to quit')
    if pr[0, 0] != y[e]:
        print('y value;', y[e])
    plt.show()
