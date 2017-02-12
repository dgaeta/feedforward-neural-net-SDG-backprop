import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

optimizer_options = {
    'epochs': 30,
    'mini_batch_size': 10, 
    'eta': 3.0 
}
import Model
model = Model.Model([784, 30, 10])

model.teach(training_data, optimizer_options, test_data)
