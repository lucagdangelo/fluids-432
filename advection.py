"""
Advection

@author: Luca D'Angelo
Feb. 25, 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# parameters
Nspace = 50
Ntime = 500
v = -0.1
dt = 1.0
dx = 1.0
alpha = v*dt/2/dx

# grid setup for FTCS method
x = np.arange(0, Nspace, dx)
f1 = np.copy(x) / Nspace
f2 = np.copy(x) / Nspace


for t in range(0, Ntime):
    for j in range(1,len(x)-1):

        # FTCS Method
        f1_cur = f1[j] - alpha*(f1[j+1]-f1[j-1])
        f1[j] = f1_cur

        # Lax-Friedrichs Method
        f2_cur = 0.5*(f2[j+1] + f2[j-1]) - alpha*(f2[j+1] - f2[j-1])
        f2[j] = f2_cur

    # plot for FTCS method
    plt.subplot(1,2,1)
    plt.plot(x, f1, color='red', marker='o')
    plt.title('FTCS Method')
    plt.xlabel('Position')
    plt.ylabel('Solution Value')
    plt.xlim([0, Nspace])
    plt.ylim([-0.1, 1])
    plt.pause(0.001)

    # plot for LF method
    plt.subplot(1,2,2)
    plt.plot(x, f2, color='purple', marker='o')
    plt.title('Lax-Friedrichs Method')
    plt.xlabel('Position')
    plt.ylabel('Solution Value')
    plt.xlim([0, Nspace])
    plt.ylim([-0.1, 1])
    plt.pause(0.001)
