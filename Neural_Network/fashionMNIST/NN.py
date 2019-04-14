### Neural Network exercise using fashion MNIST dataset ###

# based on Coursera's Machine Learning course, exercise 4, Neural Network
# Using NeuralNetwork.py, Neural_Network class, and trainer class 

#### NOTE ###
'''
This script runs optimization algorithm on FULL dataset, not batch / stochastic 
gradient descent. As a result it'll take sinificant amout of time to finish training. 
It took 30 minutes 30 seconds to train 100 iteration, on a Laptop with dual core 7th gen i7. 
'''
import pandas as pd # library to manage data
import numpy as np # matrices calculation
import matplotlib.pyplot as plt # graphing
import time # to record time took to train

#import functions as fun # bunch of functions needed for NN
#import ac # calculates accuracy  of the model after training
import fashion_data_import as fin # data importing script
import visualizer as v
from NeuralNetwork import Neural_Network
from trainer import trainer

### first, import fashion MNIST dataset

### training set
X, y = fin.data_in('train')
m, n = X.shape #(6000, 784), m for number of training examples, n for number of features
print('X: {}\ny: {}'.format(X, y))

### test set
Xt, yt = fin.data_in('test')
mt, nt = Xt.shape

### setting cloths name labels for # labels for human interpretation
human_label = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'} 
    ## here, I'm associating each neural network output label # to name of each outputs.

# display randomly selected datapoints as a picture
## methodonize it??
v.iter_img(2, X, y, human_label)
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

# show images in one plot
v.group_img(X, 10, 10)


##### Initializing Neural nets #################
print('initializing Neural Network', "."*10)
# 3 Layers total, input, 1 hidden and output layer.
input_layer_size = 784 # 28x28 input images of digits
hidden_layer_size = 500 # hidden units
output_layer_size = 10 # classify into 10 outputs, same as number of output layer's units
eps = 0.5
Net = Neural_Network(input_layer_size, output_layer_size, hidden_layer_size, eps)

##### Training NN ###########################
### Use scipy optimizer
print("initializing trainer....")
mymethod = 'L-BFGS-B' # specifying the optimization algorithm
maxiter = 10
lam =0.9 
print('training NN by ', mymethod, '.....')
t = trainer(Net, X, y)

print('Starting the training timer and training...')
start_time = time.time() # setting stopwatch for training
t.train(lam, maxiter, mymethod)
elapsed_time = time.time() - start_time

### measuring accuracy ###
accuracy = t.accuracy(Net.predict(X), y)
ptest = Net.predict(Xt)
accuracy_test = t.accuracy(ptest, yt)


### Testing set Prediction ###

minutes = elapsed_time//60
seconds = minutes%60
print("Time elapsed during training: ", minutes, "minutes", seconds, 's')
print('Training set accuracy: {}%'.format(round(accuracy, 2)))
print('Testing set accuracy: {}%'.format(round(accuracy_test, 2)))
print('Machine Learning Parameters...')
print('Sizes of the NN: input:', input_layer_size, ' hidden:', hidden_layer_size, ' output:', output_layer_size)
print('Lambda: ', lam)
print('# of iteration: ', maxiter)
print('Optimization Algorithm: ', mymethod)

##### plotting cost & accuracy history, and confusion matrix ########
## methodanize?
v.cost_acc(t.J, t.acc)
v.confused(yt, ptest, human_label)

'''
plt.figure()
plt.plot(np.arange(1, len(t.J)+1), t.J, 'b', label='cost')
plt.plot(np.arange(1, len(t.acc)+1), t.acc, 'r', label='accuracy')
plt.legend()
plt.xlabel('# of iterations')
plt.ylabel('Cost & Accuracy')
plt.grid(True)
plt.show()
'''
