# Line of best fit
# The equation of a line is: y = mx + c
# any point along x, you plug x into that equation, find the values of m and c and you have the answer to y.

# what is m?
# m = slope (best fit slope in our case)
# => m = (((mean of x) * (mean of y)) - (mean of (x*y))) / (mean of x)^2 - (mean of x^2)

# what is c?
# c = y-intercept
# => c = (mean of y) - m * (mean of x)

# let's translate it into code!

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


m, c = best_fit_slope_and_intercept(xs, ys)
print(m, c)


# create regression line using y=mx+c
regression_line = [(m*x)+c for x in xs]



# could use to make a prediction
predict_x = 8
predict_y = (m*predict_x)+c


plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
