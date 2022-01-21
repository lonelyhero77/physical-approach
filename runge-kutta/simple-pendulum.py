import numpy as np
import matplotlib.pyplot as plt

#Variable Settings
l = 1. #m
g = 10. # m/s^2
angvel0 = 0
ang0 = 1.

# Time Settings
tmax = 1000.
tstep = 0.001
t = np.linspace(0., tmax, int(tmax/tstep))

X0 = np.array([ang0, angvel0])
dt = t[1] - t[0]
nt = int(tmax/dt)
ti = np.linspace(0., nt * dt, nt)

def derivate(X, t):
    # Remember np.sin needs Angle in radians!!!!
    return np.array([X[1], -g/l * np.sin(X[0] * np.pi / 180.)])

def Euler(func, X0, t):
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        X[i+1] = X[i] + func(X[i], t[i]) * dt
    return X

def RK4(func, X0, t):
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], t[i])
        k2 = func(X[i] + dt/2. * k1, t[i] + dt/2.)
        k3 = func(X[i] + dt/2. * k2, t[i] + dt/2.)
        k4 = func(X[i] + dt * k3, t[i] + dt)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

X_euler = Euler(derivate, X0, ti)
x_euler, y_euler = X_euler[:, 0], X_euler[:, 1]

X_RK4 = RK4(derivate, X0, ti)
x_RK4, y_RK4 = X_RK4[:, 0], X_RK4[:, 1]

plt.figure()
plt.plot(ti, x_euler, color="violet", label="Euler")
plt.plot(ti, x_RK4, color="limegreen", label="RK4")
plt.grid()
plt.xlabel("t")
plt.ylabel("theta")
plt.legend()
plt.show()
