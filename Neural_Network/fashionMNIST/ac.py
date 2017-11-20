import numpy as np
def accuracy(p, y):
    comp = p == y
    comp_o_zero = comp.astype(float)
    accuracy = comp_o_zero.mean()*100
    return accuracy
