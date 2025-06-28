import numpy as np
from itertools import combinations
from numba import jit, njit, prange



def basis_set_generator(tot_sites, N_e_up, N_e_down):
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
            basis.append(up_state + down_state)
    
    return np.array(basis)


# @jit(nopython=True, parallel=True)
def creation_operator(state, site):
    
    res_state = state.copy()
    
    if state[site] == 1:
        return 0, None
    
    sign = 1
    for i in prange(site):
        if state[i] == 1:
            sign = sign * -1
    
    res_state[site] = 1
    return sign, res_state


# @jit(nopython=True, parallel=True)
def annihilation_operator(state, site):
    
    res_state = state.copy()
    
    if state[site] == 0:
        return 0, None
    
    sign = 1
    for i in prange(site):
        if state[i] == 1:
            sign = sign * -1
    
    res_state[site] = 0
    return sign, res_state


# @jit(nopython=True, parallel=True)
def hopping_operator(state, create_site, destroy_site):

    if (state[create_site] == 1) and (state[destroy_site] == 0):
        return 0, None

    sign_destroy, res_state = annihilation_operator(state, destroy_site)
    
    sign_create, res_state = creation_operator(res_state, create_site)

    return (sign_create * sign_destroy), res_state


# @jit(nopython=True, parallel=True)
def state_idx_mapping(basis_set):
    return {tuple(state) : i for i, state in enumerate(basis_set)}


# @jit(nopython=True, parallel=True)
def hamiltonian_matrix_generator(basis_set, tot_sites, J_11, J_1, J_33, J_3, U):
    dim = len(basis_set)
    hamiltonian = np.zeros((dim, dim), dtype=np.float64)
    state_idx_dict = state_idx_mapping(basis_set)

    for i, state in enumerate(basis_set):
        for j in prange(tot_sites):
            for k in [0, tot_sites]:
                l = j + k
                # print(l)

                if state[l] == 0:
                    continue

                if (l + 1) < (tot_sites + k): # J_11 & J_1 terms
                    if state[l + 1] == 0:
                        destroy_site = l
                        create_site = l + 1
                        # print(initial_position, target_position)
                        sign, res_state = hopping_operator(state, create_site, destroy_site)
                        # print(sign, res_state)

                        # print(f"{state} ---- {destroy_site} ---- {create_site} ---- {sign} ---- {res_state}")

                        if sign == 0:
                            continue

                        res_state_idx = state_idx_dict[tuple(res_state)]
                        # print(res_state_idx)

                        if l % 2 == 0:
                            hamiltonian[res_state_idx, i] += J_11 * sign
                        else:
                            hamiltonian[res_state_idx, i] += J_1 * sign
                
                if l - 1 >= k: # Hermitian conjugate of J_11 & J_1 terms
                    if state[l - 1] == 0:
                        destroy_site = l
                        create_site = l - 1
                        # print(initial_position, target_position)
                        sign, res_state = hopping_operator(state, create_site, destroy_site)
                        # print(sign, res_state)

                        # print(f"{state} ---- {destroy_site} ---- {create_site} ---- {sign} ---- {res_state}")

                        if sign == 0:
                            continue

                        res_state_idx = state_idx_dict[tuple(res_state)]
                        # print(res_state_idx)

                        if l % 2 == 0:
                            hamiltonian[res_state_idx, i] += J_1 * sign
                        else:
                            hamiltonian[res_state_idx, i] += J_11 * sign
                
                if (l + 3) < (tot_sites + k): # J_33 & J_3 terms
                    if state[l + 3] == 0:
                        destroy_site = l
                        create_site = l + 3
                        # print(initial_position, target_position)
                        sign, res_state = hopping_operator(state, create_site, destroy_site)
                        # print(sign, res_state)

                        # print(f"{state} ---- {destroy_site} ---- {create_site} ---- {sign} ---- {res_state}")

                        if sign == 0:
                            continue

                        res_state_idx = state_idx_dict[tuple(res_state)]
                        # print(res_state_idx)

                        if l % 2 == 0:
                            hamiltonian[res_state_idx, i] += J_33 * sign
                        else:
                            hamiltonian[res_state_idx, i] += J_3 * sign
                
                if (l - 3) >= k: # Hermitian conjugate of J_33 & J_3 terms
                    if state[l - 3] == 0:
                        destroy_site = l
                        create_site = l - 3
                        # print(initial_position, target_position)
                        sign, res_state = hopping_operator(state, create_site, destroy_site)
                        # print(sign, res_state)

                        # print(f"{state} ---- {destroy_site} ---- {create_site} ---- {sign} ---- {res_state}")

                        if sign == 0:
                            continue

                        res_state_idx = state_idx_dict[tuple(res_state)]
                        # print(res_state_idx)

                        if l % 2 == 0:
                            hamiltonian[res_state_idx, i] += J_3 * sign
                        else:
                            hamiltonian[res_state_idx, i] += J_33 * sign
    
    for i, state in enumerate(basis_set):
        for j in prange(tot_sites):
            if state[j] == state[j + tot_sites] and state[j] == 1:
                hamiltonian[i, i] += U
    
    return hamiltonian


def normalize(state):
    return state / np.linalg.norm(state)
