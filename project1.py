# mass action kinetics model of HIV with T cells and monocytes
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# state variables (not initial conditions; those are defined after the diffEqs)

# Drug variables
pen = 0     # level of penetration inhibitor (Enfuvirtide) in the body, normalized 0 to 1
pi = 0      # level of protease inhibitor (Ritonavir) in the body, normalized 0 to 1
nrti = 0    # level of NRTI (Tenofovir) in the body, normalized 0 to 1

# T cell variables

T_u = 1000   # initial, T cells, uninfected; there is only one uninfected strain

T_l_wt = 0     # T cells, latent infected, wildtype
T_l_pi = 0     # T cells, latent infected, protease-inhibitor resistant strain
T_l_nrti = 0   # T cells, latent infected, NRTI resistant strain
T_l_pen = 0    # T cells, latent infected, penetration-resistant strain
T_l_tot = T_l_wt + T_l_pi + T_l_nrti + T_l_pen

T_a_wt = 0     # T cells, active infected, wildtype
T_a_pi = 0     # T cells, active infected, protease-inhibitor resistant strain
T_a_nrti = 0   # T cells, active infected, NRTI resistant strain
T_a_pen = 0    # T cells, active infected, penetration-resistant strain

T_a_tot = T_a_wt + T_a_pi + T_a_nrti + T_a_pen

T_tot = T_u + T_l_tot + T_a_tot

V_wt = 0       # HIV particles, wildtype
V_pi = 0       # HIV particles, protease-inhibitor resistant strain
V_nrti = 0     # HIV particles, NRTI resistant strain
V_pen = 0      # HIV particles, penetration-resistant strain

# T cell variables
P_t_u = 20		# new production, uninfected T cells
r_t = 3		    # proliferation, T cells
T_max = 1500	# max number of T cells

mu_tu = 0.1     # rate of death, T cells, uninfected
mu_tl = 0.25	# rate of death, T cells, latently infected
mu_ta_wt = 0.25	# rate of death, T cells, actively infected with wildtype HIV

# kinetic constants
k_wt = 0.1	    # rate of infection for uninfected wildtype T cells with wildtype HIV (T_u to T_l_wt)
k_pen = 0.1	    # rate of infection for uninfected wildtype T cells with penetration resistant strain (T_u to T_l_pen)
k_pi = 0.1	    # rate of infection for uninfected wildtype T cells with protease-inhibitor resistant strain (T_u to T_l_pi)
k_nrti = 0.1    # rate of infection for uninfected wildtype T cells with NRTI resistant strain (T_u to T_l_nrti)

kt2 = 2.4e-4 	# rate of activation for latent T cells (L to I)
km1 = 0		  	# rate of infection for uninfected monocytes (M_u to M_l)
km2 = 0         # rate of activation for latent monocytes (M_u to M_l)

# virus variables
N_wt = 3.6e3		# of wildtype HIV particles produced by lysing a T cell
mu_v = 5		# rate of death, free virus

# time variables
t0 = 0
tf = 3650

# solve the system dy/dt = f(y, t)
def f(y, t):
    T_u = y[0]          # dt T cells, uninfected

    T_l_wt = y[1]       # dt T cells, latently infected, wildtype
    T_l_pen = y[2]      # dt T cells, latently infected, penetration-resistant strain
    T_l_pi = y[3]       # dt T cells, latently infected, protease-inhibitor resistant strain
    T_l_nrti = y[4]     # dt T cells, latently infected, NRTI resistant strain

    T_a_wt = y[5]       # dt T cells, actively infected, wildtype
    T_a_pen = y[6]      # dt T cells, actively infected, penetration-resistant strain
    T_a_pi = y[7]       # dt T cells, actively infected, protease-inhibitor resistant strain
    T_a_nrti = y[8]     # dt T cells, actively infected, NRTI resistant strain

    V_wt = y[9]         # dt HIV particles, wildtype
    V_pen = y[10]       # dt HIV particles, penetration-resistant strain
    V_pi = y[11]        # dt HIV particles, protease-inhibitor resistant strain
    V_nrti = y[12]      # dt HIV particles, NRTI resistant strain

    # differential equations

    # f0 = P_t_u + r_t * T_u * (1 - ( T_tot ) / T_max ) - mu_tu * T_u - k_wt * V_wt * T_u
    # f1 = k_wt * V * T_u  - mu_tl * T_l - kt2 * T_l
    # f2 = kt2 * T_l - mu_ta * T_a
    # f3 = N_t * mu_ta * T_a + - k_wt * V * T_u - mu_v * V

    # T_u; dt T cells, uninfected
    f0 = P_t_u + ( r_t * T_u * (1 - ( T_tot ) / T_max ) ) - ( k_wt * V_wt * T_u ) - ( k_nrti * V_nrti * T_u ) - ( k_pi * V_pi * T_u ) - ( k_pen * V_pen * T_u )  - ( mu_tu * T_u )

    # T_l_wt; dt T cells, latently infected, wildtype
    f1 =

    # T_l_pen; dt T cells, latently infected, penetration-resistant strain
    f2 =

    # T_l_pi; dt T cells, latently infected, protease-inhibitor resistant strain
    f3 =

    # T_l_nrti; dt T cells, latently infected, NRTI resistant strain
    f4 =

    # T_a_wt; dt T cells, actively infected, wildtype
    f5 =

    # T_a_pen; dt T cells, actively infected, penetration-resistant strain
    f6 =

    # T_a_pi; dt T cells, actively infected, protease-inhibitor resistant strain
    f7 =

    # T_a_nrti; dt T cells, actively infected, NRTI resistant strain
    f8 =

    # V_wt; dt HIV particles, wildtype
    f9 = ( N_wt * mu_ta_wt * T_a_wt ) - ( k_wt * V_wt * T_u )

    # V_pen; dt HIV particles, penetration-resistant strain
    f10 =

    # V_pi; dt HIV particles, protease-inhibitor resistant strain
    f11 =

    # V_nrti; dt HIV particles, NRTI resistant strain
    f12 =

return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

# initial conditions
T_u0 = 1000
T_l0 = 0
T_a0 = 0
V0 = 500
y0 = [T_u0, T_l0, T_a0, V0]
t = np.linspace(0, 400, 401)   # time grid

# solve the ODEs

soln = odeint(f, y0, t)
T_u = soln[:, 0]
T_l = soln[:, 1]
T_a = soln[:, 2]
V = soln[:, 3]

# plot results
plt.plot(t, T_u, label='T cells, uninfected', color=('b'))
plt.plot(t, T_l, label='T cells, latently infected', color=('g'))
plt.plot(t, T_a, label='T cells, actively infected', color=('c'))
plt.plot(t, V, label='Viral load', color=('r'))
plt.xlabel('Time (days)')
plt.ylabel('Cell count')
plt.title('White cell count & viral load against time')
plt.legend()
plt.show()
