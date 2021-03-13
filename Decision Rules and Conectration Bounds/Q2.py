import numpy as np
import matplotlib.pyplot as plt

# A
sigma = np.array([[1, 0], [0, 2]])
mu0 = np.array([[0], [0]])
mu1 = np.array([[1], [1]])
plt.contour([mu0, mu1], sigma)
