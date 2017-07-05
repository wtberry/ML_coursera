import numpy as np
class Function:
    
    ### feature normalization ####################
    
    
    def featureNormalize(self, X):
        '''
        performs normalization on each feature
        '''
        self.X = X
        mu = numpy.mean(X, axis=0)
        sigma = np.std(X, axis=0)
    
        def nor(a):
            return (a-mu)/sigma
        X_norm = np.apply_along_axis(nor, 1, X)
        return X_norm, mu, sigma
    
    
    
    ### cost function #############################
    
    
    def computeCostMulti(self, X, y, theta):
        '''
        compute the cost function for each iteration
        '''
    
        self.X = X
        self.y = y
        self.theta = theta
        m = y.size # some variables
        J = 9
    
        dif = (X.dot(theta)) - y
    
        J = 1/(2*m) * np.transpose(dif).dot(dif) # vectorization of cost func.
        return J
    
    
    
    ### gradient descent ###########################
    
    
    def gradientDescentMulti(self, X, y, theta, alpha, num_iters):
        '''
        performs batch grad descdent on given parameters for num_liter times
        '''
        self.X = X
        self.y = y
        self.theta = theta
        self.alpha = alpha
        self.num_iters = num_iters
        m = y.size # some variables
        J_history = np.zeros((num_liters, 1))
    
        for i in range(num_liters):
            predictions = X.dot(theta)
    
            theta[0] = theta[0] - alpha * (1/m) * (predictions - y).sum()
            theta[1] = theta[1] - alpha * (1/m) * ((predictions - y)*X[:,1]).sum()
            theta[2] = theta[2] - alpha * (1/m) * ((predictions - y)*X[:,2]).sum()
            theta[3] = theta[3] - alpha * (1/m) * ((predictions - y)*X[:,3]).sum()
            theta[4] = theta[4] - alpha * (1/m) * ((predictions - y)*X[:,4]).sum()
    
            J_history[i] = computeCostMulti(X, y, theta)
            if i%10 == 0:
                print(J_history[i], num_iters-i)
    
        return theta, J_history
