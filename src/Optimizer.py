#### Libraries
# Standard library
import random

# First-party library
import FeedForward
import Learner

# Third-party libraries
import numpy as np

class Optimizer: 

  """
  epochs, mini_batch_size, eta are all HYPER-PARAMETERs
  'HYPER' is used to distinguish from the parameters (weights and bias) that 
  are learnt by our learning algorithm (SGD).

  optimzer_options = {
      epochs, 
      mini_batch_size,
      eta
  }
  """
  @staticmethod
  def sgd(model, training_data, optimizer_options, test_data=None):
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

    epochs = optimizer_options['epochs']
    mini_batch_size = optimizer_options['mini_batch_size']
    eta = optimizer_options['eta']

    if test_data: 
      n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs):
      random.shuffle(training_data)
      mini_batches = [
        training_data[k:k+mini_batch_size]
              for k in xrange(0, n, mini_batch_size)
      ]
      for mini_batch in mini_batches:
        Optimizer.update_mini_batch_vectorized(model, mini_batch, eta)

      ### Useful for tracking progress, but slows 
      ### things down substantially
      if test_data:
        print "Epoch {0}: {1} / {2}".format(
            j, Optimizer.evaluate(model, test_data), n_test) ## in terms of accuracy 
      else:
        print "Epoch {0} complete".format(j)
      
      '''
      It may be interesting to return the weights and bias without applying the 
      changes directly to the model. This could allow us to 'simulate' learning.

      It may be interesting to things like, 'does ordering of learning mater'?
      '''
      # return weights, bias 


  @staticmethod
  def update_mini_batch_vectorized(model, mini_batch, eta):
    
    # Initialize the gradient vectors to 0's'
    # (nabla is the vector calculus symbol to represent the gradient)
    nabla_b = [np.zeros(b.shape) for b in model.biases_matrix]
    nabla_w = [np.zeros(w.shape) for w in model.weights_tensor]

    # For each mini_batch, run back_prop to get the changes we should make.
    # i.e. tell us the error at the output level 
    for x, y in mini_batch:
      # update the gradients 

      '''
      feedforward_result_dictionary = 
      \{ activations, zs \}
      '''
      feedforward_result_dictionary = FeedForward.FeedForward.withInputVector(
          model.weights_tensor,
          model.biases_matrix,
          x
      )

      '''
      Calculate error at the output layer
      '''
      activations = feedforward_result_dictionary['activations']
      zs = feedforward_result_dictionary['zs']
      
      delta_output_layer = Optimizer.cost_derivative(activations[-1], y) * \
        FeedForward.FeedForward.sigmoid_prime(zs[-1])

      delta_nabla_b, delta_nabla_w = Learner.Learner.backprop(
        model.weights_tensor,
        model.biases_matrix,
        delta_output_layer,
        activations, 
        zs
      )

      # Apply the changes 
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]


    #### apply a single step of gradient descent ####
    model.weights_tensor = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(model.weights_tensor, nabla_w)]
    model.biases_matrix = [b-(eta/len(mini_batch))*nb
                for b, nb in zip(model.biases_matrix, nabla_b)]        

@staticmethod
def update_mini_batch_matrixed(model, mini_batches, eta):
  
  # Initialize the gradient vectors to 0's'
  # (nabla is the vector calculus symbol to represent the gradient)
  nabla_b = [np.zeros(b.shape) for b in model.biases_matrix]
  nabla_w = [np.zeros(w.shape) for w in model.weights_tensor]

  zs = model.weights_tensor.transpose() * mini_batches

  # For each mini_batch, run back_prop to get the changes we should make.
  # i.e. tell us the error at the output level 
  for x, y in mini_batch:
    # update the gradients 

    '''
    feedforward_result_dictionary = 
    \{ activations, zs \}
    '''
    feedforward_result_dictionary = FeedForward.FeedForward.withInputVector(
        model.weights_tensor,
        model.biases_matrix,
        x
    )

    '''
    Calculate error at the output layer
    '''
    activations = feedforward_result_dictionary['activations']
    zs = feedforward_result_dictionary['zs']
    
    delta_output_layer = Optimizer.cost_derivative(activations[-1], y) * \
      FeedForward.FeedForward.sigmoid_prime(zs[-1])

    delta_nabla_b, delta_nabla_w = Learner.Learner.backprop(
      model.weights_tensor,
      model.biases_matrix,
      delta_output_layer,
      activations, 
      zs
    )

    # Apply the changes 
    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]


  #### apply a single step of gradient descent ####
  model.weights_tensor = [w-(eta/len(mini_batch))*nw
                  for w, nw in zip(model.weights_tensor, nabla_w)]
  model.biases_matrix = [b-(eta/len(mini_batch))*nb
              for b, nb in zip(model.biases_matrix, nabla_b)]             


  '''
    What do you want to be the measure of error, what is the (cost / objective) function?
    This function is the derivative of that functions.
  '''
  @staticmethod
  def cost_derivative(output_activations, y):
    '''Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations.'''
    return (output_activations-y)

  @staticmethod
  def evaluate(model, test_data):
    """
    @returns the number of test inputs for which the neural
    network outputs the correct result. 
    @remarks Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation.
    """
    test_results = [(np.argmax(FeedForward.FeedForward.withInputVector( model.weights_tensor, model.biases_matrix, x )['activations'][2]), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

