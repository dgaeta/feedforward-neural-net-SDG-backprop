This repository contains code samples that were taken from the book on ["Neural Networks
and Deep Learning"](http://neuralnetworksanddeeplearning.com) by mneilsen.

I have refactored and continue to refactor the code samples into modular software components
that will make it more stable to change components of the neural network and 
add more sophisticated additions to the neural network (such as generalization components 
or changing the model from feed-forward to recurrent).

Refactoring to this point in time:
* refactored the design into 4 main modules: Model, Optimizer, Learner, FeedForward. 
* Model: how the network is shaped, how the weights tensor is shaped, how the bias matrix is shaped.
* FeedForward: this is an abstraction for the model, the model doesn't need to know how the output is created. (we can imagine 
that a different module could be Recurrent or Convulational)
* Optimizer: the method by which we calculate error (currently uses Stochastic Gradient Descent)
* Learner: an algorithm that determines how to use the error function to change the network's behavior (currently uses backprop) 

The code is written for Python 2.6 or 2.7. Michal Daniel Dobrzanski
has a repository for Python 3
[here](https://github.com/MichalDanielDobrzanski/DeepLearningPython35). 

The program `src/network3.py` uses version 0.6 or 0.7 of the Theano
library.  It needs modification for compatibility with later versions
of the library.  I will not be making such modifications.
