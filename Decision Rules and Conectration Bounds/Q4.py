import math

import numpy as np
import matplotlib.pyplot as plt


def hoeffding(n, epsilon):
    y = 2.0 * np.exp(-2 * epsilon * epsilon * n)
    return y


if __name__ == '__main__':
    N = 200000
    n = 20
    data = np.random.binomial(n=1, p=0.5, size=(N, n))
    avgs = np.mean(data, axis=1)

    eps = np.linspace(0, 1, 50)
    empProbs = np.zeros(50)
    for i in range(50):
        sum = 0
        for avg in avgs:
            if abs(avg - 0.5) > eps[i]:
                sum += 1
        empProbs[i] = sum / N

    hoeffdings = np.array([hoeffding(n, eps[i]) for i in range(50)])
    plt.plot(eps, empProbs, c='r')
    plt.plot(eps, hoeffdings, c='b')
    plt.xlabel("Epsilons")
    text = f"""N = {N:}   n = {n:}"""
    plt.title(text)
    plt.legend(["Empirical Probability", "Hoeffdings' bound"])
    plt.show()
