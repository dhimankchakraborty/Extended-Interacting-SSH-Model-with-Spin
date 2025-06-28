import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import process_time
from functions import *



tot_sites = 8
filing_factor = 1
N_e_up = 4
N_e_down = N_e_up
J_11 = 0.1
J_1 = 2
J_33 = 0.1
J_3 = 1.5
U = 0

start_time = process_time()


e_val_arr, e_vec_ps_arr = simulate_system(tot_sites, N_e_up, N_e_down, J_11, J_1, J_33, J_3, U)
dim = len(e_val_arr)
print(dim)


e_vec_ps_prob_arr = e_vec_ps_arr ** 2
topo_state_ps_idx_arr = [dim // 2, (dim // 2) - 1]
topo_state_factor_01 = 0.005
topo_state_factor_02 = 0.1

for i in range(dim):
    if (e_vec_ps_prob_arr[i][0] * topo_state_factor_01 > e_vec_ps_prob_arr[i][1]) and (e_vec_ps_prob_arr[i][tot_sites - 1] * topo_state_factor_01 > e_vec_ps_prob_arr[i][tot_sites - 2]) and (e_vec_ps_prob_arr[i][0] * topo_state_factor_02 > e_vec_ps_prob_arr[i][(tot_sites // 2) - 1]) and (e_vec_ps_prob_arr[i][tot_sites - 1] * topo_state_factor_02 > e_vec_ps_prob_arr[i][tot_sites // 2]):
        if (e_vec_ps_prob_arr[i][2] * topo_state_factor_01 > e_vec_ps_prob_arr[i][1]) and (e_vec_ps_prob_arr[i][tot_sites - 3] * topo_state_factor_01 > e_vec_ps_prob_arr[i][tot_sites - 2]):
            topo_state_ps_idx_arr.append(i)


print(len(topo_state_ps_idx_arr))
print(topo_state_ps_idx_arr)

end_time = process_time()
cpu_time = end_time - start_time
print(f"CPU time taken: {cpu_time}")


sites_pos = np.arange(tot_sites)
# plt.rcParams['text.usetex'] = True
for i in range(len(topo_state_ps_idx_arr)):
    idx = topo_state_ps_idx_arr[i]
    plt.plot(sites_pos, e_vec_ps_prob_arr[idx], label=f'{idx}, Energy: {e_val_arr[idx]}')
    # plt.ylim(0, 1)
    plt.axhline(0)
    plt.title(f"Probability Density at Sites \n$J_{{11}}$: {J_11}, $J_1$: {J_1}, $J_3$: {J_3}, $J_{{33}}$: {J_33} & $U$: {U} \n Filing Factor: {filing_factor}")
    plt.ylabel("Probability Density for Particles at Sites($|\\psi|^2$)")
    plt.xlabel("Sites")
    plt.legend(loc="upper center")
    plt.grid()
    plt.show()
