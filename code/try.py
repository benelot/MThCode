import utilities_train as utrain
import utilities_figures as ufig
import utilities_various as uvar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sig = utrain.coupled_oscillator(t_length=4, fs=512, small_weights=True)

id_ = 'artificial_wd_strong_coupling'
ufig.plot_weights(id_)

utrain.least_squares('ID07', [32, 7], 10, 256)


# def pendulum_sim2(init_vals, t, omega=2*np.pi/3,
#     L=1, m=1, b=.5, g=9.81, F_mag=10):
#     theta1 = [init_vals['theta_1']]
#     theta2 = [init_vals['theta_2']]
#     zeta = [init_vals['zeta']]
#     dt = t[1] - t[0]
#     for i, t_ in enumerate(t[:-1]):
#         next_theta1 = theta1[-1] + theta2[-1] * dt
#         next_zeta = zeta[-1] + omega*dt
#         next_theta2 = theta2[-1] + (
#             F_mag/(m*L)*np.cos(zeta[-1]) - b/m * theta2[-1] -
#             g/L * np.sin(next_theta1)) * dt
#         theta1.append(next_theta1)
#         theta2.append(next_theta2)
#         zeta.append(next_zeta)
#     return np.stack([theta1, theta2, zeta])
#
#
# init_vals = {'theta_1': np.pi/2,
#              'theta_2': 0,
#              'zeta': 0}
#
# t_max = 1000
# dt = 0.01
# t = np.linspace(0, t_max, int(t_max/dt))
# vals = pendulum_sim2(init_vals, t)
# plt.plot(vals[0,:])
# plt.plot(vals[1,:])
#
# test = 1