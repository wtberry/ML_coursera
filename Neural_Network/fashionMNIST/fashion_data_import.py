'''
This script import fashion MNIST dataset as pandas dataframe, and convert it 
to numpy arrays. 
It then separate it into feature values and labels, and return each separately.
Enter argument as either fashion_mnist_test.csv or fashion_mnist_train.csv
'''

import numpy as np
import pandas as pd

def data_in(file_in):
    '''
    arg as either 'train' or 'test'
    AS STRING
    and return X, y in this order
    '''
    PATH = 'data' 
    printing = 'Importing ' + file_in + 'ing dataset as pandas dataframe...'
    print(printing)

    if file_in == 'train':
        data = pd.read_csv(PATH + "/mnist_train.csv")

    elif file_in == 'test':
        data = pd.read_csv(PATH + "/mnist_test.csv")


    ## converting the dataframe into np array (matrix?)
    data_array = data.values
    m = data_array.shape[0]
    ## assigning X and y for data and label
    X = data_array[:, 1:]
    y = data_array[:, 0].reshape(m, 1)

    return X, y
