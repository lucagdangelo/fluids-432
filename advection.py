"""
Advection equation solver comparing FTCS and Lax-Friedrichs methods.

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

# grid setup
x = np.arange(0, Nspace, dx)
f1 = np.copy(x) / Nspace
f2 = np.copy(x) / Nspace

# plotting setup
plt.ion()
fig1 = plt.figure(1)

# plot setup for D1
plt.subplot(1,2,1)
plot1, = plt.plot(x, f1, color='red', marker='o')
plt.title('FTCS Method')
plt.xlabel('Position')
plt.ylabel('Solution Value')
plt.xlim([0, Nspace])
plt.ylim([-0.1, 1])

# plot setup for D2
plt.subplot(1,2,2)
plot2, = plt.plot(x, f2, color='purple', marker='o')
plt.title('Lax-Friedrichs Method')
plt.xlabel('Position')
plt.xlim([0, Nspace])
plt.ylim([-0.1, 1])

fig1.canvas.draw()

for t in range(0, Ntime):
    for j in range(1,len(x)-1):

        # FTCS Method
        f1_cur = f1[j] - alpha*(f1[j+1]-f1[j-1])
        f1[j] = f1_cur

        # Lax-Friedrichs Method
        f2_cur = 0.5*(f2[j+1] + f2[j-1]) - alpha*(f2[j+1] - f2[j-1])
        f2[j] = f2_cur

    # update plot for FTCS Method
    plot1.set_ydata(f1)

    # update plot for LF Method
    plot2.set_ydata(f2)

    fig1.canvas.draw()
    plt.pause(0.001)
