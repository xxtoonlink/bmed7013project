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

T_wt = 0     # T cells, infected, wildtype
T_pi = 0     # T cells, infected, protease-inhibitor resistant strain
T_nrti = 0   # T cells, infected, NRTI resistant strain
T_pen = 0    # T cells, infected, penetration-resistant strain

T_tot = T_wt + T_pi + T_nrti + T_pen

V_wt = 0       # HIV particles, wildtype HIV
V_nrti = 0     # HIV particles, NRTI resistant strain
V_pen = 0      # HIV particles, penetration-inhibitor resistant strain
V_pi = 0       # HIV particles, protease-inhibitor resistant strain

k_nrti = 0.1    # rate of infection for uninfected wildtype T cells with NRTI resistant strain (T_u to T_l_nrti)

# T cell variables
s_t = 20		# new production, fresh T cells
r_t = 3		    # proliferation, T cells
T_max = 1500	# max number of T cells

mu_tu = 0.1      # rate of death, T cells, uninfected
mu_wt = 0.25	 # rate of death, T cells, infected with wildtype HIV
mu_nrti = 0.25   # rate of death, T cells, infected with NRTI resistant HIV
mu_pen = 0.25    # rate of death, T cells, infected with penetration-inhibitor resistant HIV
mu_pi = 0.25     # rate of death, T cells, infected with protease-inhibitor resistantHIV

# kinetic constants
k_wt = 0.1	    # rate of infection for T cells with wildtype HIV (T_u to T_wt)
k_nrti = 0.1    # rate of infection for T cells with NRTI resistant strain (T_u to T_nrti)
k_pen = 0.1	    # rate of infection for T cells with penetration-inhibitor resistant strain (T_u to T_pen)
k_pi = 0.1	    # rate of infection for T cells with protease-inhibitor resistant strain (T_u to T_pi)

# virus variables
N_wt = 3.6e3	    # of wildtype HIV particles produced by lysing a T cell
N_nrti = 3e3        # of NRTI-resistant HIV particles produced by lysing a T cell
N_pen = 3e3         # of penetration inhibitor-resistant HIV particles produced by lysing a T cell
N_pi = 3e3          # of protease-inhibitor-resistant HIV particles produced by lysing a T cell
mu_v = 5		    # rate of death, free virus
mut_nrti = 0.001    # rate of mutation, wildtype to NRTI resistant HIV
mut_pen = 0.001     # rate of mutation, wildtype to penetration-inhibitor resistant HIV
mut_pi = 0.001      # rate of mutation, wildtype to protease-inhibitor resistant HIV

# solve the system dy/dt = f(y, t)
def f(y, t):
    T_u = y[0]        # dt T cells, uninfected
    T_wt = y[1]       # dt T cells, infected, wildtype
    T_nrti = y[2]     # dt T cells, infected, NRTI resistant strain
    T_pen = y[3]      # dt T cells, infected, penetration-resistant strain
    T_pi = y[4]       # dt T cells, infected, protease-inhibitor resistant strain
    V_wt = y[5]       # dt HIV particles, wildtype
    V_nrti = y[6]    # dt HIV particles, NRTI resistant strain
    V_pen = y[7]     # dt HIV particles, penetration-resistant strain
    V_pi = y[8]      # dt HIV particles, protease-inhibitor resistant strain

    # differential equations

    eqs = np.zeros((9))

    # T_u; dt T cells, uninfected
    eqs[0] = s_t + ( r_t * T_u * ( 1 - ( T_tot / T_max ) )
    - ( k_wt * V_wt * T_u ) * (1 - nrti) * (1 - pen) * (1 - pi)
    - ( k_nrti * V_nrti * T_u ) * (1 - pen) * (1 - pi)
    - ( k_pi * V_pi * T_u ) * (1 - nrti) * (1 - pen)
    - ( k_pen * V_pen * T_u ) * (1 - nrti) * (1 - pi)
    - ( mu_tu * T_u )

    # T_wt; dt T cells, infected, wildtype
    eqs[1] = ( k_wt * V_wt * T_u ) * (1 - nrti) * (1 - pen) * (1 - pi) - ( N_wt * mu_wt * T_wt )

    # T_nrti; dt T cells, infected, NRTI resistant strain
    eqs[2] = ( k_nrti * V_nrti * T_u ) * (1 - pen) * (1 - pi) - ( mu_nrti * T_nrti )

    # T_pen; dt T cells, infected, penetration-resistant strain
    eqs[3] = ( k_pen * V_pen * T_u ) * (1 - nrti) * (1 - pi) - ( mu_pen * T_pen )

    # T_pi; dt T cells, infected, protease-inhibitor resistant strain
    eqs[4] = ( k_pi * V_pi * T_u ) * (1 - nrti) * (1 - pen) - ( mu_pi * T_pi )

    # V_wt; dt HIV particles, wildtype
    eqs[5] = ( ( N_wt * mu_wt * T_wt ) - ( k_wt * V_wt * T_u ) ) * (1 - nrti) * (1 - pen) * (1 - pi) - ( mu_v * V_wt )
    - ( mut_nrti * V_wt ) - ( mut_pen * V_wt ) - ( mut_pi * V_wt ) - ( mut_f * V_wt )

    # V_nrti; dt HIV particles, NRTI resistant strain
    eqs[6] = ( ( N_nrti * mu_nrti * T_nrti ) - ( k_nrti * V_nrti * T_u ) ) * (1 - pen) * (1 - pi) - ( mu_v * V_nrti ) + ( mut_nrti * V_wt )

    # V_pen; dt HIV particles, penetration-resistant strain
    eqs[7] = ( ( N_pen * mu_pen * T_pen ) - ( k_pen * V_pen * T_u ) ) * (1 - nrti) * (1 - pi) - ( mu_v * V_pen ) + ( mut_pen * V_wt )

    # V_pi; dt HIV particles, protease-inhibitor resistant strain
    eqs[8] = ( ( N_pi * mu_pi * T_pi ) - ( k_pi * V_pi * T_u ) ) * (1 - nrti) * (1 - pen) - ( mu_v * V_pi ) + ( mut_pi * V_wt )

    return eqs

# initial conditions
T_u_0 = 1
T_wt_0 = 0
T_nrti_0 = 0
T_pen_0 = 0
T_pi_0 = 0
V_wt_0 = 1
V_nrti_0 = 0
V_pen_0 = 0
V_pi_0 = 0

y0 = [T_u_0, T_wt_0, T_nrti_0, T_pen_0, T_pi_0, V_wt_0, V_nrti_0, V_pen_0, V_pi_0]

t = np.linspace(0, 365, 366)   # time grid

# solve the ODEs

soln = odeint(f, y0, t)

T_u = soln[:, 0]
T_wt = soln[:, 1]
T_nrti = soln[:, 2]
T_pen = soln[:, 3]
T_pi = soln[:, 4]
V_wt = soln[:, 5]
V_nrti = soln[:, 6]
V_pen = soln[:, 7]
V_pi = soln[:, 8]

# plot results
plt.plot(t, T_u, label='T cells, uninfected', color=('b'))
plt.plot(t, T_wt, label='T cells, infected, wildtype', color=('c'))
plt.plot(t, V_wt, label='HIV particles, wildtype', color=('r'))
plt.xlabel('Time (days)')
plt.ylabel('Cell count')
plt.title('T cell count & viral load against time')
plt.legend()
plt.show()
