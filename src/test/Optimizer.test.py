import sys
sys.path.append('../')
import Optimizer
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

Optimizer.Optimizer.update_mini_batch_vectorized(model1, mini_batches[0], eta)

Optimizer.Optimizer.update_mini_batch_matrixed(model2, mini_batches, eta)