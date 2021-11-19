import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import statistics
from numpy.linalg import inv
import random


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = []
    file = open(filename, encoding="utf-8")
    read = list(csv.reader(file))
    file.close
    header = read[0]
    read = read[1:]
    i = 1;
    for i in read:
        if i == "IDNO":
            continue
        r = (i[1:])  # r -> row
        r = list(map(float, r))
        dataset.append((r))

    return np.array(dataset)


def print_stats(dataset, col):
    size = len(dataset)
    print(size)
    y_ht = sum(float(i[1]) for i in dataset) / size
    print("{:.2f}".format(y_ht))
    arr_y = (i[1] for i in dataset)
    s_d = statistics.stdev(arr_y)
    print("{:.2f}".format(s_d))

    return


def regression(dataset, cols, betas):
    ttl = 0  # ttl - total
    for i in dataset:
        sum1 = 0
        for col in cols:
            sum1 += float(i[col]) * float(betas[cols.index(col) + 1])
        ttl += math.pow(((betas[0]) + (sum1) - float(i[0])), 2)
    s = ttl / len(dataset)
    return s


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat num by m+1 array
        cols    - a list of feat indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    num = len(dataset)
    for b_i in range(len(betas)):
        ttl1 = 0
        for i in dataset:
            sum1 = 0
            for col in cols:
                sum1 += (i[col]) * (betas[cols.index(col) + 1])
            if b_i == 0:
                y = 1.0
            else:
                y = y = i[cols[b_i - 1]]

            ttl1 += ((betas[0]) + (sum1) - (i[0])) * (y)
            # print(len(grads))
        ttl1 = 2 * (ttl1 / num)
        grads.append(ttl1)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat num by m+1 array
        cols    - a list of feat indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    temp = betas
    mse_grad = gradient_descent(dataset, cols, betas)
    for t in range(1, T + 1):
        String = ""
        for i in range(len(temp)):
            temp[i] = temp[i] - eta * mse_grad[i]
        String = str(t) + " " + str(round(regression(dataset, cols, temp), 2)) + " "
        for j in range(len(temp)):
            String += " " + str("{:.2f}".format(temp[j]))
        mse_grad = gradient_descent(dataset, cols, temp)

    return None


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat num by m+1 array
        cols    - a list of feat indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    y = []
    x = []
    for r in dataset:
        x.append([1])
        y.append(r[0])
    for r in range(len(dataset)):
        for col in range(len(cols)):
            x[r].append(dataset[r][cols[col]])
    x = np.array(x)
    y = np.array(y)
    x_in = inv(np.dot(np.transpose(x), x))
    f = np.dot(np.dot(inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)

    mse = regression(dataset, cols, f)

    tupp = []
    tupp.append(mse)
    for element in f:
        tupp.append(element)
    tupp = tuple(tupp)

    return tupp


def predict(dataset, cols, features):
    betas = compute_betas(dataset, cols)[1:]
    X = [1]
    for feat in features:
        X.append(feat)
    X = np.array(X)
    res = np.dot(X, betas)
    return res


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (num,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (num,2) - linear one first, followed by quadratic.
    """
    num = len(X)
    l_inear = []
    for i in range(num):
        temp = []
        temp.append(X[i][0])
        l_inear.append(temp)
    mean = 0
    for j in range(len(l_inear)):
        temp = betas[0] + betas[1] * l_inear[j][0] + np.random.normal(mean, sigma)
        l_inear[j].insert(0, temp)
    q_uadratic = []
    for i in range(num):
        temp = []
        temp.append(X[i][0])
        q_uadratic.append(temp)
    for j in range(len(q_uadratic)):
        temp = alphas[0] + alphas[1] * (math.pow(q_uadratic[j][0], 2)) + np.random.normal(mean, sigma)
        q_uadratic[j].insert(0, temp)
    return l_inear, q_uadratic


from math import e


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
    X = []
    for i in range(1000):
        X.append([random.randint(-100, 100)])
    X = np.array(X)
    betas = np.array([random.random(), random.random()])
    alphas = np.array([random.random(), random.random()])
    sigmas = [(10 ** i) for i in range(-4, 6)]
    mse_linear = []
    mse_quadratic = []
    for sigma in sigmas:
        linear, quadratic = synthetic_datasets(betas, alphas, X, sigma)
        mse_linear.append(compute_betas(linear, [1])[0])
        mse_quadratic.append(compute_betas(quadratic, [1])[0])
    plt.figure()
    plt.plot(sigmas, mse_linear, label="MSE - Linear Dataset", marker="o")
    plt.plot(sigmas, mse_quadratic, label="MSE - Quadratic Dataset", marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sigmas")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('mse.pdf')

    plt.show()


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()






