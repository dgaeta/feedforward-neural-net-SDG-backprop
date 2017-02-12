import sys
sys.path.append('../')
import Optimizer
import FeedForward
import mnist_loader
import random

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

optimizer_options = {
    'epochs': 30,
    'mini_batch_size': 10, 
    'eta': 3.0 
}
import Model
model1 = Model.Model([784, 30, 10])
model2 = Model.Model([784, 30, 10])

n = len(training_data)
random.shuffle(training_data)
mini_batch_size = 10
eta = 3.0
mini_batches = [
    training_data[k:k+mini_batch_size]
        for k in xrange(0, n, mini_batch_size)
]
print len(mini_batches)

feedforward_vector_results = FeedForward.FeedForward.withInputVector(model1.weights_tensor, model1.biases_matrix, mini_batches[0])
feedforward_matrix_results = FeedForward.FeedForward.withInputMatrix(model2.weights_tensor, model2.biases_matrix, mini_batches
)