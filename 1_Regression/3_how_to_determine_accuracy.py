# how do we calculate how good of a fit our best fit line is?

# R Squared Theory = coefficient of determination
# calculated using squared error

# What is squared error?
# the error is the distance betwee the point and the best fit line
# squared error means we are only dealing with positive values
# use square over absolute values because we want to penalise outliers
# squared error is the standard

# how do we calculate R Squared (SE)?
# r^2 = 1 - ((SE of the y-hat-line) / SE of the mean of the y)

# eg. if r^2 is 0.8
# in order for r^2 to be 0.8, ((SE of the y-hat-line) / SE of the mean of the y) would have to be 0.2
# further, this could mean that (SE of the y-hat-line) could be 2 and SE of the mean of the y could be 10
# this is pretty good and means that this data is pretty linear

# eg. if r^2 is 0.3, ((SE of the y-hat-line) / SE of the mean of the y) would have to be 0.7
# further, this could mean that (SE of the y-hat-line) could be 7 and SE of the mean of the y could be 10
# this is worse, since they are a lot closer together


# let's translate r^2 into code!
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# dummy data for x values and y values
# make sure to use numpy arrays
# make sure data type default is numpy float64
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# calculate best fit slope and y-intercept
def best_fit_slope_and_intercept(xs, ys):
    # NOTE: Order of operations: PEMDAS!!! ARGH!
    # => m = (((mean of x) * (mean of y)) - (mean of (x*y))) / (mean of x)^2 - (mean of x^2)
    m = (
        ((mean(xs) * mean(ys)) - mean(xs * ys)) /
        ((mean(xs)**2) - (mean(xs**2)))
    )

    # => c = (mean of y) - m * (mean of x)
    c = (mean(ys) - m*mean(xs))

    return m, c


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_of_the_regression_line = squared_error(ys_orig, ys_line)
    squared_error_y_mean_line = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_of_the_regression_line / squared_error_y_mean_line)


m, c = best_fit_slope_and_intercept(xs, ys)
print(m, c)


# create regression line using y=mx+c
regression_line = [(m*x)+c for x in xs]



# could use to make a prediction
predict_x = 8
predict_y = (m*predict_x)+c

# calculation of how good the fit of the best fit line is using r^2:
r_squared = coefficient_of_determination(ys, regression_line)
print("R^2: ", r_squared)


plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
