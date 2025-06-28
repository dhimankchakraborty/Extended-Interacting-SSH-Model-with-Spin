import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import process_time
from functions import *



tot_sites = 6
filing_factor = 1
N_e_up = tot_sites // 2
N_e_down = N_e_up

J_11 = 0.1
J_1 = 1
J_33 = 0.1

# J_3 = 1.75
U = 10

step_no = 101
J_3_start = 0.25
J_3_end = 1.75
J_3_arr = np.linspace(J_3_start, J_3_end, step_no)

E_val_arr = []
# E_vec_ps_arr = []

for J_3 in J_3_arr:
    print(tot_sites, N_e_up, N_e_down, J_11, J_1, J_33, J_3, U)
    e_val_arr, e_vec_ps_arr = simulate_system(tot_sites, N_e_up, N_e_down, J_11, J_1, J_33, J_3, U)
    E_val_arr.append(e_val_arr)
    print(J_3 - J_1)
print()

E_val_arr = np.array(E_val_arr).transpose()

delta_J_3_J_1 = J_3_arr - J_1

for e_arr in E_val_arr:
    plt.plot(delta_J_3_J_1, e_arr)
plt.grid()
plt.show()
