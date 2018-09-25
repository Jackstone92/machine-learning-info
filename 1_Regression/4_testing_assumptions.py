# How to test our assumptions



from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')



# dummy data for x values and y values
# make sure to use numpy arrays
# make sure data type default is numpy float64
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def create_dataset(how_much, variance, step=2, correlation=False):
    val = 1 # first value for y
    ys = []
    for i in range(how_much):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'positive':
            val += step
        elif correlation and correlation == 'negative':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


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


# create data set using def
xs, ys = create_dataset(40, 40, 2, correlation='positive')


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
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()
