'''
This script import fashion MNIST dataset as pandas dataframe, and convert it 
to numpy arrays. 
It then separate it into feature values and labels, and return each separately.
Enter argument as either fashion_mnist_test.csv or fashion_mnist_train.csv
'''

import numpy as np
import pandas as pd

class data(object):
    '''
    Object represent the training and testing data
    - read in the csv file
    - convert them into np.array
    - maybe preprocessing in the future??
    '''
    def __init__(self, mode='mnist'):
        '''
        mode: str, fashionMNIST or regular MNIST, defaults to fashion
        '''
        self.PATH = 'data'
        self.X, self.y = None, None
        self.Xt, self.yt = None, None

        print("Importing {} dataset as pandas dataframe...".format(mode))
        if mode.lower() == 'fashion':
            train = pd.read_csv(self.PATH + "/fashion-mnist_train.csv")
            test = pd.read_csv(self.PATH + "/fashion-mnist_test.csv")
        elif mode.lower() == 'mnist':
            train = pd.read_csv(self.PATH + "/mnist_train.csv")
            test = pd.read_csv(self.PATH + "/mnist_test.csv")

        ## now convert into numpy arr
        train_arr = train.values
        test_arr = test.values

        self.X = train_arr[:, 1:]
        self.y = train_arr[:, 0].reshape(-1, 1)

        self.Xt = test_arr[:, 1:]
        self.yt = test_arr[:, 0].reshape(-1, 1)
    
    def getData(self, which='train'):
        if which.lower()=='train': return self.X, self.y  
        elif which.lower()=='test': return self.Xt, self.yt
