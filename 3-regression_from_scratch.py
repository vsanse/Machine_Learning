from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

#plt.scatter(xs,ys)
#plt.show()

def best_fit_slope_and_intercept(xs,ys):
    m = ((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2)-mean(xs**2))
    return m,( mean(ys) - (m * mean(xs)))

def squared_error(ys_orignal,ys_line  ):
    return sum((ys_line-ys_orignal)**2)
def r_squared(ys_orignal,ys_line):
    y_mean_line = [mean(ys_orignal) for y in ys_orignal]
    squared_error_regression = squared_error(ys_orignal,ys_line)
    squared_error_y_mean = squared_error(ys_orignal,y_mean_line)
    return 1 - (squared_error_regression/squared_error_y_mean) 

m, b= best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

accuracy = r_squared(ys, regression_line)

print(accuracy) 

plt.scatter(predict_x,predict_y,color='g')
plt.scatter(xs,ys)
plt.plot(regression_line)
plt.show()
