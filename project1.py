# mass action kinetics model of HIV with T cells and monocytes
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# state variables (not initial conditions; those are defined after the diffEqs)

# Drug variables
pen = 0     # level of penetration inhibitor (Enfuvirtide) in the body, normalized 0 to 1
pi = 0      # level of protease inhibitor (Ritonavir) in the body, normalized 0 to 1
nrti = 0    # level of NRTI (Tenofovir) in the body, normalized 0 to 1
res_all = (1 - nrti) * (1 - pen) * (1 - pi)
res_nrti = (1 - pen) * (1 - pi)
res_pen = (1 - nrti) * (1 - pi)
res_pi = (1 - nrti) * (1 - pen)

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

V_wt = 0       # HIV particles, wildtype HIV
V_nrti = 0     # HIV particles, NRTI resistant strain
V_pen = 0      # HIV particles, penetration-inhibitor resistant strain
V_pi = 0       # HIV particles, protease-inhibitor resistant strain

k_nrti = 0.1    # rate of infection for uninfected wildtype T cells with NRTI resistant strain (T_u to T_l_nrti)

# T cell variables
P_t_u = 20		# new production, uninfected T cells
r_t = 3		    # proliferation, T cells
T_max = 1500	# max number of T cells

mu_tu = 0.1         # rate of death, T cells, uninfected
mu_tl = 0.25    	# rate of death, T cells, latently infected
mu_ta_wt = 0.25	    # rate of death, T cells, actively infected with wildtype HIV
mu_ta_nrti = 0.25   # rate of death, T cells, actively infected with NRTI resistant HIV
mu_ta_pen = 0.25    # rate of death, T cells, actively infected with penetration-inhibitor resistant HIV
mu_ta_pi = 0.25     # rate of death, T cells, actively infected with protease-inhibitor resistantHIV

# kinetic constants
k_wt = 0.1	    # rate of infection for uninfected wildtype T cells with wildtype HIV (T_u to T_l_wt)
k_nrti = 0.1    # rate of infection for uninfected wildtype T cells with NRTI resistant strain (T_u to T_l_nrti)
k_pen = 0.1	    # rate of infection for uninfected wildtype T cells with penetration-inhibitor resistant strain (T_u to T_l_pen)
k_pi = 0.1	    # rate of infection for uninfected wildtype T cells with protease-inhibitor resistant strain (T_u to T_l_pi)

c_tl_wt = 2.4e-4 	# rate of activation for latent to active T cells infected with wildtype HIV
c_tl_nrti = 2.4e-4 	# rate of activation for latent to active T cells infected with NRTI resistant HIV
c_tl_pen = 2.4e-4 	# rate of activation for latent to active T cells infected with penetration-inhibitor resistant HIV
c_tl_pi = 2.4e-4 	# rate of activation for latent to active T cells infected with protease-inhibitor resistantHIV

# virus variables
N_wt = 3.6e3	    # of wildtype HIV particles produced by lysing a T cell
N_nrti = 3e3        # of NRTI-resistant HIV particles produced by lysing a T cell
N_pen = 3e3         # of penetration inhibitor-resistant HIV particles produced by lysing a T cell
N_pi = 3e3          # of protease-inhibitor-resistant HIV particles produced by lysing a T cell
mu_v = 5		    # rate of death, free virus
mut_nrti = 0.001    # rate of mutation, wildtype to NRTI resistant HIV
mut_pen = 0.001     # rate of mutation, wildtype to penetration-inhibitor resistant HIV
mut_pi = 0.001      # rate of mutation, wildtype to protease-inhibitor resistant HIV
mut_f = 0.01        # rate of mutation, wildtype to failed substrain

# solve the system dy/dt = f(y, t)
def f(y, t):
    T_u = y[0]          # dt T cells, uninfected
    T_l_wt = y[1]       # dt T cells, latently infected, wildtype
    T_l_nrti = y[2]     # dt T cells, latently infected, NRTI resistant strain
    T_l_pen = y[3]      # dt T cells, latently infected, penetration-resistant strain
    T_l_pi = y[4]       # dt T cells, latently infected, protease-inhibitor resistant strain
    T_a_wt = y[5]       # dt T cells, actively infected, wildtype
    T_a_nrti = y[6]     # dt T cells, actively infected, NRTI resistant strain
    T_a_pen = y[7]      # dt T cells, actively infected, penetration-resistant strain
    T_a_pi = y[8]       # dt T cells, actively infected, protease-inhibitor resistant strain
    V_wt = y[9]         # dt HIV particles, wildtype
    V_nrti = y[10]      # dt HIV particles, NRTI resistant strain
    V_pen = y[11]       # dt HIV particles, penetration-resistant strain
    V_pi = y[12]        # dt HIV particles, protease-inhibitor resistant strain

    # differential equations

    eqs = np.zeros((13))

    # T_u; dt T cells, uninfected
    eqs[0] = P_t_u + ( r_t * T_u * ( 1 - ( T_tot ) / T_max ) )
    - ( k_wt * V_wt * T_u ) * res_all
    - ( k_nrti * V_nrti * T_u ) * res_nrti
    - ( k_pi * V_pi * T_u ) * res_pi
    - ( k_pen * V_pen * T_u ) * res_pen
    - ( mu_tu * T_u )

    # T_l_wt; dt T cells, latently infected, wildtype
    eqs[1] = ( k_wt * V_wt * T_u ) * (1 - nrti) * (1 - pen) * (1 - pi) - ( c_tl_wt * T_l_wt )

    # T_l_nrti; dt T cells, latently infected, NRTI resistant strain
    eqs[2] = ( k_nrti * V_nrti * T_u ) * (1 - pen) * (1 - pi) - ( c_tl_nrti * T_l_nrti )

    # T_l_pen; dt T cells, latently infected, penetration-resistant strain
    eqs[3] = ( k_pen * V_pen * T_u ) * (1 - nrti) * (1 - pi) - ( c_tl_pen * T_l_pen )

    # T_l_pi; dt T cells, latently infected, protease-inhibitor resistant strain
    eqs[4] = ( k_pi * V_pi * T_u ) * (1 - nrti) * (1 - pen) - ( c_tl_pi * T_l_pi )

    # T_a_wt; dt T cells, actively infected, wildtype
    eqs[5] = ( c_tl_wt * T_l_wt) - ( N_wt * mu_ta_wt * T_a_wt )

    # T_a_nrti; dt T cells, actively infected, NRTI resistant strain
    eqs[6] = ( c_tl_nrti * T_l_nrti) - ( N_nrti * mu_ta_nrti * T_a_nrti )

    # T_a_pen; dt T cells, actively infected, penetration-resistant strain
    eqs[7] = ( c_tl_pen * T_l_pen) - ( N_pen * mu_ta_pen * T_a_pen )

    # T_a_pi; dt T cells, actively infected, protease-inhibitor resistant strain
    eqs[8] = ( c_tl_pi * T_l_pi) - ( N_pi * mu_ta_pi * T_a_pi )

    # V_wt; dt HIV particles, wildtype
    eqs[9] = ( ( N_wt * mu_ta_wt * T_a_wt ) - ( k_wt * V_wt * T_u ) ) * (1 - nrti) * (1 - pen) * (1 - pi) - ( mu_v * V_wt )
    - ( mut_nrti * V_wt ) - ( mut_pen * V_wt ) - ( mut_pi * V_wt ) - ( mut_f * V_wt )

    # V_nrti; dt HIV particles, NRTI resistant strain
    eqs[10] = ( ( N_nrti * mu_ta_nrti * T_a_nrti ) - ( k_nrti * V_nrti * T_u ) ) * (1 - pen) * (1 - pi) - ( mu_v * V_nrti )
    + ( mut_nrti * V_wt )

    # V_pen; dt HIV particles, penetration-resistant strain
    eqs[11] = ( ( N_pen * mu_ta_pen * T_a_pen ) - ( k_pen * V_pen * T_u ) ) * (1 - nrti) * (1 - pi) - ( mu_v * V_pen )
    + ( mut_pen * V_wt )

    # V_pi; dt HIV particles, protease-inhibitor resistant strain
    eqs[12] = ( ( N_pi * mu_ta_pi * T_a_pi ) - ( k_pi * V_pi * T_u ) ) * (1 - nrti) * (1 - pen) - ( mu_v * V_pi )
    + ( mut_pi * V_wt )

    return eqs

# initial conditions
T_u_0 = 1000
T_l_wt_0 = 0
T_l_nrti_0 = 0
T_l_pen_0 = 0
T_l_pi_0 = 0
T_a_wt_0 = 0
T_a_nrti_0 = 0
T_a_pen_0 = 0
T_a_pi_0 = 0
V_wt_0 = 500
V_nrti_0 = 0
V_pen_0 = 0
V_pi_0 = 0

y0 = [T_u_0, T_l_wt_0, T_l_nrti_0, T_l_pen_0, T_l_pi_0, T_a_wt_0, T_a_nrti_0, T_a_pen_0, T_a_pi_0, V_wt_0, V_nrti_0, V_pen_0, V_pi_0]

t = np.linspace(0, 3650, 3651)   # time grid

# solve the ODEs

soln = odeint(f, y0, t)

T_u = soln[:, 0]
T_l_wt = soln[:, 1]
T_l_nrti = soln[:, 2]
T_l_pen = soln[:, 3]
T_l_pi = soln[:, 4]
T_a_wt = soln[:, 5]
T_a_nrti = soln[:, 6]
T_a_pen = soln[:, 7]
T_a_pi = soln[:, 8]
V_wt = soln[:, 9]
V_nrti = soln[:, 10]
V_pen = soln[:, 11]
V_pi = soln[:, 12]

# plot results
plt.plot(t, T_u, label='T cells, uninfected', color=('b'))
plt.plot(t, T_l_wt, label='T cells, latently infected, wildtype', color=('g'))
plt.plot(t, T_a_wt, label='T cells, actively infected, wildtype', color=('c'))
plt.plot(t, V_wt, label='HIV particles, wildtype', color=('r'))
plt.xlabel('Time (days)')
plt.ylabel('Cell count')
plt.title('White cell count & viral load against time')
plt.legend()
plt.show()
