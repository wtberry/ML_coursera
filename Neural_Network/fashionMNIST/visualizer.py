'''
methods for visualizing data, such as
- cost & accuracy history
- training / testing set images
- confusion matrix
'''
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def iter_img(num_img, X, y, human_label):
    for i in range(num_img):
        sample = np.random.choice(X.shape[0]) # random index number 
        pixel = X[sample, :].reshape(-1, int(np.sqrt(X.shape[1])))
        y_val = y[sample, 0]
        lab = 'label: ' + str(y_val)
        plt.imshow(pixel, cmap='Greys')
        plt.xlabel('{} {}'.format(lab, str(human_label[y_val])))
        plt.axis('on')
        plt.show()

def group_img(data, rows, columns):
    '''
    plot given # of dataset's images in one plot
    '''
    amount = rows*columns
    '''
    do this in the other for loop

    image = np.zeros((amount, 28, 28)) # 3D container to store images
    for i in range(amount):
        image[i] = data[i, :].reshape(-1, 28) # reshape and fill in the container with image

    '''
    fig = plt.figure(figsize=(rows, columns))

    for i in range(amount):
        img = data[i, :].reshape(-1, 28)
        ax = fig.add_subplot(rows, columns, i+1)
        plt.imshow(img, cmap='magma')
        plt.axis('off')
        plt.sca(ax)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def cost_acc(cost, acc):
    '''
    Plot cost and acc history over training iteration
    make it into plot several graphs in one plot in future in for loop??
        - accept dictionary of list of data
        - xlabel name
    '''

    plt.figure()
    plt.plot(np.arange(1, len(cost)+1), cost, 'b', label='cost')
    plt.plot(np.arange(1, len(acc)+1), acc, 'r', label='accuracy')
    plt.legend()
    plt.xlabel('# of iterations')
    plt.ylabel('Cost & Accuracy')
    plt.grid(True)
    plt.show()

def confused(label, predicted_label, human_label):
    '''
    plot confusion matrix, based on given prediction and label
    '''
    cm = confusion_matrix(label, predicted_label)
    axis_font = {'size':13, 'color':'black'} # font dictionary for axis
    data_font = {'size':10, 'color':'white', 'weight':'heavy'}
    classNames = {human_label[i] for i in range(10)} # list of label names
    tick_marks = np.arange(len(classNames))

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix", fontdict=axis_font)
    plt.xlabel("Predicted Label", fontdict=axis_font)
    plt.ylabel("True Label", fontdict=axis_font)
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(cm[i, j]), fontdict=data_font, \
                horizontalalignment='center',verticalalignment='center')

    plt.show()