# What are kernels?

# In the real world, it is highly likely you are not going to get linearly separable data
# Might have to add n dimensions before we can find linearly separable data
# Can use kernels to use n dimensions in order to make svm work with real-world data
# Kernel is a similarity function. It takes two inputs and outputs their similarity
# Can use kernels to augment or add to support vector machine

# Kernels are done using inner product (or dot product)

# Let's start at the end and work backwards
# say we have an unknown feature set x
# classification => y = sign((vector w) . (vector x) + b)
# w . x is going to return a scalar value -> therefore modifying x space to Z space (unkown dimension space) won't cause any problems

# Two major constraints:
# Requirement that y sub i * (x sub i . w + b) -1 >= 0
# We can exchange x sub i with z sub i
# Other constraint:
# W = Sum of alpha sub i . (y sub i) . (x sub i)

# every interaction is a dot product!
# L = sum of alpha sub i - 1/2(sum from ij . alpha sub j . y sub i . y sub j . (x sub i . x sub j))


# Making a kernel from scratch:
# K(x, (x prime)) = z . (z prime)

# what is z?
# z is some sort of function that is being applied to its x counterpart
# z = function(x) and z prime = function(x prime)
# K is the inner product between z and z prime
# have to use the same function for z and z prime!
# Kernel is usually denoted by phi (o with vertical line down middle)


# eg. using 2D feature set
# feature set X = [x1, x2]
# Let's say we want to take ourselves out to z space -> must convert to second order polynomial:
# => Z = [1, x1, x2, x1^2, x2^2, x1*x2]
# and
# Z prime = [1, x1 prime, x2 prime, x1 prime^2, x2 prime^2, x1 prime * x2 prime]
# Therefore
# K(x, x prime) = Z . Z prime
# => [1*1, x1 * x1 prime, x2 * x2 prime, x1^2 * x1 prime^2, x2^2 * x2 prime^2, x1*x1 prime * x2 * x2 prime]
# => [1 + (x1*x prime1) + (x2*x2 prime) + (x1^2*x1 prime^2) + (x2^2*x2 prime^2) + (x1*x1 prime*x2*x2 prime)]


# Can use the polynomial kernel:
# K(x, x prime) = (1 + x . x prime)^p
# => (1 + x1*x1 prime + ... xn * xn prime)^p


# Another kernel is RBF (Radio Basis Function) - more complex and can't really conceptualise it
# K(x, x prime) = exponential(-gamma * (absolute value of x - x prime)^2)
# and an exponential function is: exp(x) = e^x
