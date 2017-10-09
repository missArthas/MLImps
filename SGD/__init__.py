# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def momentum(x_start, step, g, discount=0.7):
    #plt.contourf(X, Y, Z, 10, alpha=0.6, cmap=plt.cm.hot)
    C = plt.contour(X, Y, Z, 10, colors='black', linewidth=0.5)
    plt.plot(0,0,'bo')
    #plt.clabel(C, inline=True, fontsize=10)


    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad * step
        x -= pre_grad

        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;

    plt.show()
    return x

def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]
def g(x):
    return np.array([2 * x[0], 100 * x[1]])

def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(15, 7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            print(arr[i:i + 2, 0], arr[i:i + 2, 1])
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1])
    plt.show()


def gd(x_start, step, g):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot


def momentum(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad * step
        x -= pre_grad

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot


def nesterov(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * 0.7 + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

if __name__ == "__main__":
    xi = np.linspace(-200, 200, 1000)
    yi = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xi, yi)
    Z = X * X + 50 * Y * Y

    #contour(X, Y, Z)
    # res, x_arr = gd([150, 75], 0.016, g)
    # contour(X, Y, Z, x_arr)
    #
    # res, x_arr = gd([150, 75], 0.019, g)
    # contour(X,Y,Z, x_arr)
    #
    # res, x_arr = gd([150, 75], 0.02, g)
    # contour(X, Y, Z, x_arr)

    res, x_arr = momentum([150, 75], 0.016, g)
    contour(X, Y, Z, x_arr)

    res, x_arr = nesterov([150, 75], 0.012, g)
    contour(X, Y, Z, x_arr)


