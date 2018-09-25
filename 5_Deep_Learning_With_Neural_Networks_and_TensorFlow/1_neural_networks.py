# A neural network is a network of neurons
# Biological neural network - dendrytes, nucleus, axon, axon terminal, synapse
# Nothing in the neural network model uses biological terms, but we model after it!

# Artificial neural network - input data (eg. x1, x2, x3) that is uniquely weighted (w1, w2, w3) and then summed together
# a neuron either fires or doesn't fire -> passed through threshold function (or step function) and if threshold is passed, it fires (either results in a 0 or a 1)
    # threshold function is usually a sigmoid function (more of an S shape than step) -> activation function
    # y = f(vector x, vector y)

# Deep neural network (more than 1 hidden layers)
    # input, hidden layer 1, hidden layer 2, output

# datasets:
# imageNet for images
# wikipedia datadump for text data
# commonCrawl <-



# what is tensorflow?
# a matrix manipulation library
# a tensor is an array-like object
# tensorflow is basically just functions on arrays
# python is inherently a slow language - line by line etc -> inefficient!
# tensorflow: define model in abstract terms, when you're ready, you run the session and you then get the result

# tensorflow basics:
import tensorflow as tf

# construct graph
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1, x2)

print(result)
# => Tensor("Mul:0", shape=(), dtype=int32)

# # run a session -> nothing happens until we run a session
# sess = tf.Session()
# print(sess.run(result))
# # => Tensor("Mul:0", shape=(), dtype=int32)
# # => 30
#
# # close the session
# sess.close()


# better way to do it (will automatically close when done):
with tf.Session() as sess:
    output = sess.run(result)
    print(output)

# saves to python variables too!
print(output)













#
