### Feedforward Neural Network on FashionMNIST

- This is the neural network built from scrach without machine leanring libraries. Only libraries used was...
* NumPy: Matrix operations
* SciPy: sigmoid function
* Pandas: data preprocessing
* Matplotlib: visualization
* Time: measures training time

[fashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, fashion version of MNIST with images from Zalando, was used.


## What does *.py scripts does...
- fashion_data_import.py: imports the datasets from data/ folder, and preprocess it into 
numpy array, and return them.

- function.py: includes all functions needed for the NN, such as activation, and cost function.

- ac.py: is a script to evaluate the trained neural network. It compares the predicted output
and the 'correct answers' from the test dataset, and calculates the accuracy and 
return the value.


- Data file is not uploaded, due to file size limit of github. 

## What about *.ipynb?
They go over basic concepts of neural network, for my own record / memory keeping.
- Bokeh Sample.ipynb
- Cost Function.ipynb
- Neural Network - FashionMNIST dataset.ipynb
- Random Ititialization & Matrix Folding.ipynb
- Sigmoid Function
