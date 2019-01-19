# Multi-variable linear regression

This is a practice project for multi-variable linear regression using a dataset collected from a Combined Cycle Power Plant over 6 years (2006-2011). It was retrieved from Machine Learning Repository of University of California, Irvine.
  
  ( http://archive.ics.uci.edu/ml/index.php )

The dataset contains 4 features, including Tempurature(T), Ambient Pressure(AP), Relative Humidity(RH), and Exhaust Vaccum(V) to predict the net hourly electrical energy output(EP). 

For detailed info for the dataset, please refer to 'linearReg/multi/CCPP_dataset/Readme.txt~'

* linear_multi.py: read the dataset, and perform feature normalization, initialize cost function, and perfom batch gradient descent.

It outputs the number of iterations left as well as the change of cost function J while it performs the descent.

Finally it outputs the graph of cost function, final theta values and a sample prediction of PE based on the first row of the dataset.

* Linear_multi.py: main script to run training & predictions
