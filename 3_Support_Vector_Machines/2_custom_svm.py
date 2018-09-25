# Vectors

# Vector A would be notated as A with an arrow pointing right on top of it
# Vectors have magnitude and direction

# eg. [3,4] basically like cartesian coordinates
# direction is from origin (in this case [0, 0]) to vector A as an arrow
# magnitude (notated by double bars around vector A) is the same as the norm or length
# magnitude = square root of the squared constituants summed together
# => magnitude of a = squareroot(3^2 + 4^2) = 5

# dot product between two vectors
# eg. vector A = [1, 3] and vector B = [4, 2]
# to get dot product: A . B
# -> (1*4) + (3*2) = 4 + 6 = 10 (and it is a scalar value)

# in order to determine how unknown vector U should be classified if vector W that points perpendicularly towards the best separating hyperplane:
# => calculation = vector U . vector W + bias
# if that is >= 0, then it would be a positive sample (on the positive side of the best separating hyperplane)
# otherwise if it is <= 0, then it would be a negative sample (on the negative side of the best separating hyperplane)
# if it is equal to 0, then it would be on the decision boundary!

# hence, use y sub i:
# +class => (vector x sub i) . vector w + b = 1
# therefore y sub i * ((vector x sub i) . vector w + b) = 1
# -class => (vector x sub i) . vector w + b = -1
# therefore y sub i * ((vector x sub i) . vector w + b) = -1

# we want both of these equations equal to 0
# for +class => y sub i * ((vector x sub i) . vector w + b)) -1 = 0
# for -class => y sub i * ((vector x sub i) . vector w + b)) -1 = 0

# therefore calculation can be expressed as:
# y sub i * ((vector x sub i) . vector w + b)) -1

# in order to find width:
# width = ((vector x from positive class) - (vector x from negative class)) . ((vector w) / magnitude of vector w)
# => width 2 / (magnitude of vector w)
# we want to maximise width, so we want to minimise the magnitude of vector w with our constraint y sub i * ((vector x sub i) . vector w + b)) -1
# therefore for mathematic convenience, we can write it as:
# width = 1/2 * (magnitude of vector w)^2

# Legrangian statement to optimise:
# L(w, b) = (1/2 * (magnitude of vector w)^2) - sum(alpha sub i)[y sub i * ((vector x sub i) . vector w + b)) -1]
# we want to minimise w and maximise b

# need to differentiate L with respect to w, and differentiate L with respect to b:
# => diff(L) / diff(w) = vector w = sum((alpha sub i)(y sub i)(vector x sub i))
# => diff(L) / diff(b) = -(sum((alpha sub i)(y sub i))) = 0



# equation for a hyperplane: (vector X sub i) . (vector W) + b
# equation for support vectors:
# (vector X sub i) . (vector W) + b = 1 (for +class)
# (vector X sub i) . (vector W) + b = -1 (for -class)



# how do we optimise for W and B?
# minimise magnitude of vector W and maximise b

# in python: class(knownFeatures . W + b) >= 1
# will return dictionary mag = {magnitude of w: [w, b]}

# this is a convex problem
# convex line is the magnitude of vector w
# imaging dropping ball in a bowl. Will roll around until it rests at the global minimum (bottom middle)

# eg. vector w = [5, 3]
# magnitude of vector w = sqrt(5^2 + 3^3) = sqrt(34)
# if vector w = [-5, 3]
# magnitude of vector w = sqrt(34) as well!
# however, we need to be able to test for both positive and negative hyperplane slopes


# creating svm from scratch
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


# build support vector class
class Support_Vector_Machine:
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colours = {1: 'r', -1: 'b'}

        if self.visualisation:
            # matplotlib config
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # training and optimsation method
    def fit(self, data):
        # primitive optimisation -> check other resources for better algorithms
        self.data = data

        # will look like: { ||w||: [w, b]}
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        # use to get maximum and minimum data
        all_data = []
        # crude for loop
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # reset memory
        all_data = None

        # start with big steps and have decreasing steps
        # support vectors yi(xi.w+b) = 1
        # we will know when both positive and negative classes, you have a value that is close to 1
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point where it gets expensive
                      self.max_feature_value * 0.001,
        ]

        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps with b as we do with w
        b_multiple = 5

        # first element in vector w => save a lot of processing
        latest_optimum = self.max_feature_value * 10


        # begin stepping process:
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # optimised will be false until there are no more steps down => convex
            optimised = False

            while not optimised:
                # iterate through bs to get maximum
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                        self.max_feature_value * b_range_multiple,
                                        step * b_multiple):
                    for transformation in transforms:
                        # transform w be each transformation [1,1], [-1,1]... etc.
                        w_t = w * transformation
                        found_option = True # innocence until proven guilty
                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this a bit
                        for class_ in self.data:
                            for xi in self.data[class_]:
                                yi = class_
                                # constraint function: yi(xi . w + b) >= 1
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # break
                                # print(xi, ':', yi * (np.dot(w_t, xi) + b))

                        # if everything worked
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimised = True
                    print('Optimised a step.')
                else:
                    # w is currently = [5, 5]
                    # step might be 1 (scalar value)
                    # w - step = [4, 4] works programatically, but probably not mathematically
                    w = w - step

            # magnitudes
            norms = sorted([n for n in opt_dict])
            # optimal choice is smallest norm
            opt_choice = opt_dict[norms[0]]

            # set optimised values
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # modify latest_optimum
            latest_optimum = opt_choice[0][0] + step * 2

        # show values
        # for class_ in self.data:
        #     for xi in self.data[class_]:
        #         yi = class_
        #         print(xi, ':', yi * (np.dot(self.w, xi) + self.b))



    # prediction method
    def predict(self, features):
        # sign( x.w + b ) - whatever the sign of the equation is
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # plot using matplotlib
        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1], s=200, marker='*', color=self.colours[classification])

        return classification


    # purely for human visualisation - no effect on svm at all!
    def visualise(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colours[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x . w + b
        # v = x . w + b
        # we care what v is where +class = 1 and -class = -1 and decision boundary = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # use to limit our graph so data isn't on the edge
        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyperplane_x_min = datarange[0]
        hyperplane_x_max = datarange[1]

        # positive support vector hyperplane = (w.x + b) = 1
        psv1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [psv1, psv2], 'k')

        # negative support vector hyperplane = (w.x + b) = -1
        nsv1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [nsv1, nsv2], 'k')

        # decision boundary = (w.x + b) = 0
        db1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [db1, db2], 'y--')

        plt.show()



# start with simple data
data_dict = {
    -1: np.array([ [1,7],
                   [2,8],
                   [3,8] ]),
    1: np.array([  [5,1],
                   [6,-1],
                   [7,3] ])
}


svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]
]

for p in predict_us:
    svm.predict(p)


svm.visualise()
