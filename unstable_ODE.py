"""
File: unstable_ODE.py
Copyright (c) 2016 Andrew Malfavon
License: MIT
Exercise C.4
Description:
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#given du
def u_prime(u, t, const):
    return const * u

#forward euler method based on code from book
def Forward_Euler(u0, du, alpha, a, b, dt):
    n = int((b - a) / (dt))
    t = np.zeros(n + 1)
    u = np.zeros(n + 1)
    u_k = np.zeros(n + 1)
    u[0] = u0
    t[0] = 0
    u_k[0] = u0
    for i in range(n):
        t[i + 1] = t[i] + dt
        u[i + 1] = u[i] + dt * du(u[i], t[i], alpha)
        u_k[i + 1] = ((1 + alpha * dt)**(i + 1)) * u0
        #used to compare the forward euler method with the given equation
    return u, t, u_k

#plot the oscillations
def plot(u0, alpha, a, b, dt):
    x = Forward_Euler(u0, u_prime, alpha, a, b, dt)[1]
    y = Forward_Euler(u0, u_prime, alpha, a, b, dt)[0]
    plt.xlabel('t')
    plt.ylabel('u')
    plt.title('Oscillation of numerical solution')
    plt.plot(x, y)

#test the array made with the forward euler method to the array made from the given equation in the book.
def test():
    approx = Forward_Euler(1, u_prime, -1, 0, 121, 1.1)[0]
    exact = Forward_Euler(1, u_prime, -1, 0, 121, 1.1)[2]
    assert (approx.all() - exact.all()) < 1e-100