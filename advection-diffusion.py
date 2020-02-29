"""
Advection-diffusion equation solver using
the LF method for the advection term and
the implicit method for the diffusion term.

@author: Luca D'Angelo
Feb. 28, 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# parameters
Nspace = 50
Ntime = 5000
v = -0.1
dt = 0.5
dx = 0.5
D1 = 0.2
D2 = 0.05
alpha = v*dt/2/dx
beta1 = D1*dt/(dx**2)
beta2 = D2*dt/(dx**2)

# grid setup
x = np.arange(0, Nspace, dx)
f1 = np.copy(x) / Nspace
f2 = np.copy(x) / Nspace

# diffusive term with beta1
A1 = np.eye(len(x)) * (1.0 + 2.0 * beta1) + np.eye(len(x), k=1) * -beta1 + np.eye(len(x), k=-1)*-beta1
f1 = np.linalg.solve(A1,f1)

# diffusive term with beta2
A2 = np.eye(len(x)) * (1.0 + 2.0 * beta2) + np.eye(len(x), k=1) * -beta2 + np.eye(len(x), k=-1)*-beta2
f2 = np.linalg.solve(A2,f2)

# plotting setup
plt.ion()

# plot setup for D1
plt.subplot(1,2,1)
fig1 = plt.figure(1)
plot1, = plt.plot(x, f1, color='red', marker='o')
plt.title('Advection-Diffusion D=0.2')
plt.xlabel('Position')
plt.ylabel('Solution Value')
plt.xlim([0, Nspace])
plt.ylim([-0.1, 1])

# plot setup for D2
plt.subplot(1,2,2)
plot2, = plt.plot(x, f2, color='purple', marker='o')
plt.title('Advection-Diffusion D=0.05')
plt.xlabel('Position')
plt.xlim([0, Nspace])
plt.ylim([-0.1, 1])

fig1.canvas.draw()

# advection terms
for t in range(0, Ntime):
    for j in range(1,len(x)-1):

        # Lax-Friedrichs Method for advection term with beta1
        f1_cur = 0.5*(f1[j+1] + f1[j-1]) - alpha*(f1[j+1] - f1[j-1])
        f1[j] = f1_cur

        # Lax-Friedrichs Method for advection term with beta2
        f2_cur = 0.5*(f2[j+1] + f2[j-1]) - alpha*(f2[j+1] - f2[j-1])
        f2[j] = f2_cur

    # update plot for D1
    plot1.set_ydata(f1)

    # update plot for D2
    plot2.set_ydata(f2)

    # update canvas
    fig1.canvas.draw()
    plt.pause(0.001)