#### Libraries
# Third-party libraries
import numpy as np
import FeedForward

'''
@author Danny Gaeta
'''

class Learner(object):
    
  @staticmethod
  def backprop(weights_tensor, bias_matrix, delta, activations, zs):
    '''
    backpropagation algorithm

    A way to pass the blame backwards through the network to each layer, 
    and subsequently to each weight and bias.

    @returns a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  
    ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``.
    '''
    
    if not weights_tensor:
      print 'Expected key, weights_tensor, to be set on optimizer_options'
      return

    if not bias_matrix:
      print 'Expected key, bias_matrix, to be set on optimizer_options'
      return
    
    if delta.size == 0:
      print 'Expected key, delta, to be set on optimizer_options'
      return

    if (not activations):
      print 'Expected key, activations, to be set on optimizer_options'
      return
    
    if (not zs):
      print 'Expected key, zs, to be set on optimizer_options'
      return

    nabla_b = [np.zeros(b.shape) for b in bias_matrix]
    nabla_w = [np.zeros(w.shape) for w in weights_tensor]

    ### We already know the error at the output layer, 
    # no need to calculate that
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    ### It may look like we are going forward but we're actually moving backward
    num_layers = len(weights_tensor) + 1
    for l in xrange(2, num_layers):
      z = zs[-l]
      sp = FeedForward.FeedForward.sigmoid_prime(z)
      delta = np.dot(weights_tensor[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return (nabla_b, nabla_w)

