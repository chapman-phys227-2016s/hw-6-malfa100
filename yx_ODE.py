"""
File: yx_ODE.py
Copyright (c) 2016 Andrew Malfavon
License: MIT
Exercise C.3
Description: Solve an ODE using the Forward Euler method.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#ode to be solved
def y_prime(y, x):
    return 1 / (2 * (y - 1))

#forward euler method based on code from book
def Forward_Euler(prime, a, b, dx, eps = 1e-3):
    n = int((b - a) / (dx))
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    y[0] = 1 + np.sqrt(eps)
    x[0] = 0
    for i in range(n):
        x[i + 1] = x[i] + dx
        y[i + 1] = y[i] + dx * prime(y[i], x[i])
    return y, x

#solves the ode using sympy
#does not take into account initial condition
def sympy_solution():
    x = sp.Symbol('x')
    f = sp.Function('f')
    eq = 1 / (2 * (f(x) - 1))
    return sp.dsolve(sp.Eq(sp.diff(f(x)), eq))[1]#solves ode. first solution is negative so we use the second solution

#plot approximations with three step sizes and plots analytical solution
def plot(eps = 1e-3):
    x_exact = np.linspace(0, 4, 1001)
    y_exact = 1 + np.sqrt(x_exact + eps)
    x1 = Forward_Euler(y_prime, 0, 4, 1)[1]
    y1 = Forward_Euler(y_prime, 0, 4, 1)[0]
    x2 = Forward_Euler(y_prime, 0, 4, 0.25)[1]
    y2 = Forward_Euler(y_prime, 0, 4, 0.25)[0]
    x3 = Forward_Euler(y_prime, 0, 4, 0.01)[1]
    y3 = Forward_Euler(y_prime, 0, 4, 0.01)[0]
    plt.plot(x_exact, y_exact, label = 'Exact Solution')#labels are for the key in the graph
    plt.plot(x1, y1, label = 'step size = 1')
    plt.plot(x2, y2, label = 'step size = 0.25')
    plt.plot(x3, y3, label = 'step size = 0.01')
    plt.title('Approximations and exact solution of ODE')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)#creates key for graph

#test the most accurate approximation with the analytical solution
def test():
    eps = 1e-3
    exact = 1 + np.sqrt(4 + eps)
    assert (Forward_Euler(y_prime, 0, 4, 0.01)[0][-1]) - (exact) < 0.01