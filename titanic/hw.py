import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def model(t, state, r1, r2, N1, N2, sigma1, sigma2):
    x, y = state
    dxdt = r1*x*(1-x/N1-sigma1*y/N2)
    dydt = r2*y*(-1+sigma2*x/N1)
    return [dxdt, dydt]

r1 = 0.5
r2 = 0.2
N1 = 100
N2 = 200
sigma1 = 0.5
sigma2 = 0.5
t = np.linspace(0, 100, 1000)
x0 = 10
y0 = 20

sol = solve_ivp(model, [t[0], t[-1]], [x0, y0], args=(r1, r2, N1, N2, sigma1, sigma2), t_eval=t)
x = sol.y[0]
y = sol.y[1]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase space plot')
plt.show()
