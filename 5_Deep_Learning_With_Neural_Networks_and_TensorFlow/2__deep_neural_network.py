# using MNIST dataset: handwritten numbers

'''
input data -> weights -> hidden layer 1 (activation function) -> weights -> hidden layer 2 (activation function) -> weights -> output layer

feed forward neural network - data being passed straight through

compare output to intended output -> cost function eg. cross entropy (how wrong are we)

optimisation function (optimiser) -> attempt to minimise the cost (eg. AtomOptimizer... SGD, AdaGrad)

backpropagation - optimisation or going backwards and manipulating the weights

feed forward + backpropagation = epoch (or 1 cycle)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # one_hot -> will result in one final

# 10 classes of handwritten 0-9
'''
one_hot means:
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
...
'''

# set number of hidden layer nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# set number of classes
n_classes = 10

# go through batches of 100 features at a time and then go through next batch (aids RAM)
batch_size = 100

# x is the input data
x = tf.placeholder('float', [None, 784]) # ensure that shape of input data is 28x28 pixels
# y is label of that data
y = tf.placeholder('float')


# define neural network model
def neural_network_model(data):
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal(n_nodes_hl1))
    }

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal(n_nodes_hl2))
    }

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal(n_nodes_hl3))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    # model for each layer = (input_data * weights) + biases
    # level 1 sum function
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])
    # activation function
    l1 = tf.nn.relu(l1) # rectified linear

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output
