import numpy as np
from scipy.integrate import dblquad

R = 3
r = 1.5
k = 0.6
n = 5


# Интегрируем функцию f(φ, θ)
def integrand(phi, theta):
    return (R + r*np.cos(phi) + k*np.sin(n*theta))**3 * np.sin(phi) / 3


# Пределы интегрирования: phi от 0 до pi, theta от 0 до 2*pi
volume, error = dblquad(integrand, 0, 2*np.pi, lambda θ: 0, lambda θ: np.pi)
print("Объём фигуры:", volume, "err:", error)
