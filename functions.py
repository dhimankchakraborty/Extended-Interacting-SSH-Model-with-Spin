import numpy as np
import copy
import scipy as sp
from itertools import combinations
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.sparse import dok_matrix




def generator_basis_set(tot_sites, N_e_up, N_e_down):
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

    basis = []

    for up_state in spin_up_basis:
        for down_state in spin_down_basis:
            basis.append([up_state, down_state])
    
    return np.array(basis)



def hop_j_2_k(input_state, j, k, spin):
    if input_state[spin][j] == 0 or input_state[spin][k] == 1:
        return 0, None
    
    state = copy.deepcopy(input_state)
    
    sign = 1

    # Destrying the particle

    for i in range(j):
        if state[spin][i] == 1:
            sign = sign * (-1)
    
    state[spin][j] = 0

    # Create the particle

    for i in range(k):
        if state[spin][i] == 1:
            sign = sign * (-1)
    
    state[spin][k] = 1

    return sign, state



def map_state_2_idx(basis_set):
    return {tuple(np.concatenate((state[0], state[1]))) : i for i, state in enumerate(basis_set)}



def generate_hamiltonian_matrix(basis_set, dim, tot_sites, U, J_1, J_11, J_3, J_33):

    state_2_idx_mapping = map_state_2_idx(basis_set)
    H = np.zeros((dim, dim))

    for a in range(dim):
        state = basis_set[a].copy()

        # Interaction Potential Energy

        for i in range(tot_sites):
            H[a, a] += U * state[0][i] * state[1][i]
        
        # Kinetic Energy

        for s in range(2):
            for j in range(tot_sites):
                for k in [j + 1, j - 1, j + 3, j - 3]:
                    if (k >= tot_sites) or (k < 0):
                        continue

                    hopped_state_sign, hopped_state = hop_j_2_k(state, j, k, s)
                    
                    if hopped_state_sign == 0:
                        continue

                    b = state_2_idx_mapping[tuple(np.concatenate((hopped_state[0], hopped_state[1])))]

                    # # Test Print
                    # print(state[0], state[1], '------', s, '---', hopped_state[0], hopped_state[1], '---', hopped_state_sign, '--', b + 1)

                    if j % 2 == 0: # Identifies site A

                        if k == j + 1:
                            H[a, b] -= J_11 * hopped_state_sign

                        elif k == j - 1:
                            H[a, b] -= J_1 * hopped_state_sign

                        elif k == j + 3:
                            H[a, b] -= J_33 * hopped_state_sign
                        
                        else:
                            H[a, b] -= J_3 * hopped_state_sign
                    
                    else: # Identifies site B
                        if k == j + 1:
                            H[a, b] -= J_1 * hopped_state_sign

                        elif k == j - 1:
                            H[a, b] -= J_11 * hopped_state_sign

                        elif k == j + 3:
                            H[a, b] -= J_3 * hopped_state_sign
                        
                        else:
                            H[a, b] -= J_33 * hopped_state_sign
            
    return H



def generate_hamiltonian_matrix_sparse(basis_set, dim, tot_sites, U, J_1, J_11, J_3, J_33):
    state_2_idx_mapping = map_state_2_idx(basis_set)
    H = dok_matrix((dim, dim), dtype=np.float64)

    for a in range(dim):
        state = basis_set[a].copy()

        # Interaction Potential Energy
        for i in range(tot_sites):
            H[a, a] += U * state[0][i] * state[1][i]

        # Kinetic Energy
        for s in range(2):
            for j in range(tot_sites):
                for k in [j + 1, j - 1, j + 3, j - 3]:
                    if (k >= tot_sites) or (k < 0):
                        continue

                    hopped_state_sign, hopped_state = hop_j_2_k(state, j, k, s)

                    if hopped_state_sign == 0:
                        continue

                    b = state_2_idx_mapping[tuple(np.concatenate((hopped_state[0], hopped_state[1])))]

                    if j % 2 == 0:  # A site
                        if k == j + 1:
                            H[a, b] -= J_11 * hopped_state_sign
                        elif k == j - 1:
                            H[a, b] -= J_1 * hopped_state_sign
                        elif k == j + 3:
                            H[a, b] -= J_33 * hopped_state_sign
                        else:
                            H[a, b] -= J_3 * hopped_state_sign
                    else:  # B site
                        if k == j + 1:
                            H[a, b] -= J_1 * hopped_state_sign
                        elif k == j - 1:
                            H[a, b] -= J_11 * hopped_state_sign
                        elif k == j + 3:
                            H[a, b] -= J_3 * hopped_state_sign
                        else:
                            H[a, b] -= J_33 * hopped_state_sign

    return H



# @njit
# def generate_pos_sp_wf(psi, basis_set, tot_sites):
#     pos_sp_state = np.zeros((2, tot_sites))  # [spin][site]

#     for idx, amp in enumerate(psi):
#         up_state, down_state = basis_set[idx]
#         for site in prange(tot_sites):
#             pos_sp_state[0, site] += amp * up_state[site]
#             pos_sp_state[1, site] += amp * down_state[site]

#     return pos_sp_state
# @njit
# def generate_pos_sp_wf(psi, basis_set, tot_sites, dim):
#     pos_sp_state = np.zeros((2, tot_sites))  # [spin][site]


#     # for idx, amp in enumerate(psi):
#     #     up_state, down_state = basis_set[idx]
#     #     for j in prange(len(basis_set)):
#     #         pos_sp_state[0] += amp * up_state[site]
#     #         pos_sp_state[1] += amp * down_state[site]
    
#     # for idx in prange(len(basis_set)):
#     #     # up_state, down_state = basis_set[idx]
#     #     pos_sp_state[0] += psi[idx] * basis_set[idx][0]
#     #     pos_sp_state[1] += psi[idx] * basis_set[idx][1]

#     for i in prange(dim):
#         for j in prange(tot_sites):
#             pos_sp_state[0][j] += psi[i] * basis_set[i][0][j]
#             pos_sp_state[1][j] += psi[i] * basis_set[i][1][j]

#     return pos_sp_state



@njit(parallel=True)
def position_space_particle_probability(psi, basis_set, dim, tot_sites):
    probability = np.zeros((2, tot_sites))

    for j in prange(tot_sites):
        val_up = 0.0
        val_down = 0.0

        for idx in prange(dim):
            up, down = basis_set[idx]
            n_up = up[j]
            n_down = down[j]

            val_up += psi[idx] * n_up * psi[idx]
            val_down += psi[idx] * n_down * psi[idx]
        
        probability[0][j] = val_up
        probability[1][j] = val_down
    
    return probability




@njit(parallel=True)
def generate_spin_correlation(psi, basis_set, dim, tot_sites):
    C = np.zeros((tot_sites, tot_sites))

    for j in prange(tot_sites):
        for k in prange(tot_sites):
            val = 0.0
            for idx in prange(dim):
                up, down = basis_set[idx]
                sz_j = 0.5 * (up[j] - down[j])
                sz_k = 0.5 * (up[k] - down[k])
                val += psi[idx] * sz_j * sz_k * psi[idx]
            C[j, k] = val

    return C



def normalize(state):
    return state / (np.linalg.norm(state))
