# https://numerical-analysis.readthedocs.io/en/latest/ODE/ODE.html 
import numpy as np
import matplotlib.pyplot as plt

tmax = .2
# linspace = Linearly Spaced, np.linspace(start, stop, num) creates one-dimension array
"""
np.linspace(2.0, 3.0, num=5)
array([2.  , 2.25, 2.5 , 2.75, 3.  ])
"""

t = np.linspace(0., tmax, 1000)
x0, y0 = 0., 0.
vx0, vy0 = 1., 1.
g = 10.
x = vx0 * t
y = -g * t**2/2. + vy0 * t
print(x)
print(y)
dt = 0.02
X0 = np.array([0., 0., vx0, vy0])
nt = int(tmax/dt)
ti = np.linspace(0., nt * dt, nt)

def derivate(X, t):
	return np.array([X[2], X[3], 0., -g])

def Euler(func, X0, t):
	dt = t[1] - t[0]
	nt = len(t)
	X = np.zeros([nt, len(X0)])
	X[0] = X0
	for i in range(nt-1):
		X[i+1] = X[i] + func(X[i], t[i]) * dt
	return X

X_euler = Euler(derivate, X0, ti)
x_euler, y_euler = X_euler[:, 0], X_euler[:, 1]

def RK4(func, X0, t):
	dt = t[1] - t[0]
	nt = len(t)
	X = np.zeros([nt, len(X0)])
	X[0] = X0
	for i in range(nt-1):
		k1 = func(X[i], t[i])
		k2 = func(X[i] + dt/2. * k1, t[i] + dt/2.)
		k3 = func(X[i] + dt/2. * k2, t[i] + dt/2.)
		k4 = func(X[i] + dt * k3, t[i] + dt)
		X[i+1] = X[i] + dt/6. *(k1 + 2. * k2 + 2. * k3 + k4)
	return X
X_rk4 = RK4(derivate, X0, ti)
print(X_rk4)
x_rk4, y_rk4 = X_rk4[:, 0], X_rk4[:, 1]

plt.figure()
plt.plot(x, y, label = "Exact Solution")
plt.plot(x_euler, y_euler, 'or',  label = "Euler")
plt.plot(x_rk4, y_rk4, 'gs', label = "RK4")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
