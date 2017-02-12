#### Libraries

# Third-party libraries
import numpy as np

'''
@author Danny Gaeta

This is a feature of the 'model' class. 
It is a particular implementation of function, 'f(x)', applied
to the input vector.
'''

class FeedForward(object):
    '''This class implements members that compute the FeedForward step 
    of a neural network learning algorithm, in both the vectorized form 
    and the matrix form.
    ''' 

    @staticmethod
    def withInputMatrix(weights_tensor, bias_matrix, X):
        '''
        Given an input 'a' for the network, returns the corresponding 
        output*, all the method does is applies
        a'=sigma(wa+b) for each layer:
            
        @param weights - a tensor containing weight matrices
        @param bias - vector of biases
        @param a - the input layer values  

        @returns [activations_vector, weighted_inputs_vector]
        '''

        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer (called weighted input)
        for b_vector, W_matrix in zip(bias_matrix, weights_tensor):
            z = np.dot(W_matrix, activation)+b_vector
            zs.append(z)
            activation = FeedForward.sigmoid(z)
            activations.append(activation)
        
        return {'activations': activations, 'zs': zs}

    @staticmethod
    def withInputVector(weights_tensor, bias_matrix, x):
        '''
        Given an input 'a' for the network, returns the corresponding 
        output*, all the method does is applies
        a'=sigma(wa+b) for each layer:
            
        @param weights - a tensor containing weight matrices
        @param bias - vector of biases
        @param a - the input layer values  

        @returns [activations_vector, weighted_inputs_vector]
        '''

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer (called weighted input)
        for b, w in zip(bias_matrix, weights_tensor):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = FeedForward.sigmoid(z)
            activations.append(activation)
        
        return {'activations': activations, 'zs': zs}

    @staticmethod
    def sigmoid(z):
        """The sigmoid function.
        @param z - z is a vector or Numpy array

        @returns Numpy automatically applies the function sigmoid elementwise, 
        that is, in vectorized form.
        """
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return FeedForward.sigmoid(z)*(1-FeedForward.sigmoid(z))