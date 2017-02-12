"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

@remarks you can speed things up by decreasing the number of epochs, 
by decreasing the number of hidden neurons, or by using only part 
of the training data.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
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
        self.num_layers = len(sizes)
        self.sizes = sizes
        # biases are only ever used in computing the outputs from later layers
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Start at 1 cause we don't need a bias for the input layer'
        
        # Create a matrix of size 
        # y - each node has to have as many edges as there are neurons in the next layer (how many targets each neuron has)
        # x - we need y * (the number of neurons in the sending layer) (how many senders there are)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] # Start at 1 cause we don't need weights into our input layer'

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    """
    epochs, mini_batch_size, eta are all HYPER-PARAMETERs
    'HYPER' is used to distinguish from the parameters (weights and bias) that 
    are learnt by our learning algorithm (SGD).
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  

        @param training_data - is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  
        @param eta - this is the learning rate.
        @param test_data - if provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test) ## in terms of accuracy 
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        @param mini_batch - is a list of tuples ``(x, y)``.
        @param eta - the learning rate.
        """
        # (nabla is the vector calculus symbol to represent the gradient)
        # Initialize the gradient vectors to 0's'
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For each mini_batch, run back_prop to get the changes we should make.
        for x, y in mini_batch:
            # update the gradients 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # delta_x = step_size * gradient_x  (eta/len(mini_batch) is the step size)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        backpropagation algorithm

        A way to pass the blame backwards through the network to each layer, 
        and subsequently to each weight and bias. 

        @returns a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  
        ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer (called weighted input)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        @returns the number of test inputs for which the neural
        network outputs the correct result. 
        @remarks Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function.
    @param z - z is a vector or Numpy array

    @returns Numpy automatically applies the function sigmoid elementwise, 
    that is, in vectorized form.
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
