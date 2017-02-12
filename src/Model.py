#### Libraries
# Standard library
import random

# FIrst-party libraries
import Optimizer

# Third-party libraries
import numpy as np

class Model(object):

    def __init__(self, sizes, network_type='fully-connected-network'):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        if network_type == 'fully-connected-network':

            self.num_layers = len(sizes)
            self.sizes = sizes
            # biases are only ever used in computing the outputs from later layers
            self.biases_matrix = [np.random.randn(y, 1) for y in sizes[1:]] # Start at 1 cause we don't need a bias for the input layer'
            
            # Create a matrix of size 
            # y - each node has to have as many edges as there are neurons in the next layer (how many targets each neuron has)
            # x - we need y * (the number of neurons in the sending layer) (how many senders there are
            '''
            Example: 
            For a network of layers containing [784, 30, 10]

            then weights_tensor[0] = weight matrix for layer (0) to layer (1). In this 
            example, it will be a matrix of size (384, 30). 
            Where each column represents the weights for a single sending neuron.
            '''
            self.weights_tensor = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])] # Start at 1 cause we don't need weights into our input layer'

    def teach(self, training_data, optimizer_options, test_data, optimizer="SGD", ):
        '''
        Possible additional parameters to be added: 
            learner = "Backprop"
            learner_options = dictionary 
        '''
        
        '''
        Optimizer options = epochs, mini_batch_size, eta,
            test_data=None

        Learner options = eta, 
        '''
        '''optimizer_options = {
            'epochs': 30,
            'mini_batch_size': 10, 
            'eta' = 3.0
        '''
        '''
        Create an optimized learning expirience for our model. 
        '''
        if (not training_data):
            print 'Expected key, epochs, to be set on optimizer_options'
            return

        if (not ('epochs' in optimizer_options.keys())):
            print 'Expected key, epochs, to be set on optimizer_options'
            return

        if (not ('mini_batch_size' in optimizer_options.keys())):
            print 'Expected key, mini_batch_size, to be set on optimizer_options'
            return

        if (not ('eta' in optimizer_options.keys())):
            print 'Expected key, eta, to be set on optimizer_options'
            return

        optimizer_options = {'epochs': 30, 'mini_batch_size': 10, 'eta': 3.0}
        Optimizer.Optimizer.sgd(self, training_data, optimizer_options, test_data)