import numpy as np
from itertools import combinations


def basis_set_generator(tot_sites, N_e_up, N_e_down): # Checked OK
    basis = []

    spin_up_basis = []
    spin_down_basis = []

    for comb_up in combinations(range(tot_sites), N_e_up):
        state = [0] * tot_sites
        for idx in comb_up:
            state[idx] = 1
        spin_up_basis.append(state)

    for comb_down in combinations(range(tot_sites), N_e_down):
        state = [0] * tot_sites
        for idx in comb_down:
            state[idx] = 1
        spin_down_basis.append(state)
    
    spin_up_basis = np.array(spin_up_basis)
    spin_down_basis = np.array(spin_down_basis)

    for up_state in spin_up_basis:
        for down_state in spin_down_basis:
            basis.append([up_state, down_state])
    
    return np.array(basis)

