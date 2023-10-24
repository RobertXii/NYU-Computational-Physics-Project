import matplotlib.pyplot as plt
import numpy as np
from cells import Cell


nt = 300
nx = 100
v0 = 0.1
rows = nt
cols = nx
grid_cur = []

# initialize
for j in range(nt):
    grid_cur.append([])
    for i in range(nx):
        if i < nx / 2:
            rho_L = 1
            P_L = 0.8
            grid_cur[j].append(Cell(rho_L, v0, P_L))
        else:
            rho_R = 0.1
            P_R = 0.1
            grid_cur[j].append(Cell(rho_R, v0, P_R))


def find_p(cur_cell):
    cur_cell.P = (cur_cell.gamma-1)*cur_cell.U[0]*(cur_cell.U[2]/cur_cell.U[0]-0.5*(cur_cell.U[1]/cur_cell.U[0]**2))
    # print(cur_cell.P)
    return cur_cell.P


def find_lambda(cur_cell):
    cur_cell.lambda_plus = cur_cell.U[1]/cur_cell.U[0] + (cur_cell.gamma*find_p(cur_cell)/cur_cell.U[0])**0.5
    cur_cell.lambda_minus = cur_cell.U[1]/cur_cell.U[0] - (cur_cell.gamma*find_p(cur_cell)/cur_cell.U[0])**0.5
    # print(cur_cell.U)
    return cur_cell.lambda_plus, cur_cell.lambda_minus


def find_f_half(cur_cell, r_cell):
    lambda_plus, lambda_minus = find_lambda(cur_cell)
    # print(lambda_plus,lambda_minus)
    for ii in range(3):
        lambda_plus_cur = lambda_plus*cur_cell.U[ii]
        lambda_plus_r = lambda_plus*r_cell.U[ii]
        lambda_minus_cur = -lambda_minus*cur_cell.U[ii]
        lambda_minus_r = -lambda_minus*r_cell.U[ii]
        cur_cell.alpha_plus[ii] = max([0, lambda_plus_cur.real, lambda_plus_r.real])
        cur_cell.alpha_minus[ii] = max([0, lambda_minus_cur.real, lambda_minus_r.real])

    for ii in range(3):
        cur_cell.F_half[ii] = (cur_cell.alpha_plus[ii]*cur_cell.F[ii] + cur_cell.alpha_minus[ii]*r_cell.F[ii]
        - cur_cell.alpha_plus[ii]*cur_cell.alpha_minus[ii]*(r_cell.U[ii]-cur_cell.U[ii])) \
        / (cur_cell.alpha_plus[ii] + cur_cell.alpha_minus[ii])

    return cur_cell.F_half


def find_u_later(cur_cell, r_cell, l_cell, dx, dt):
    for ii in range(3):
        du = -(find_f_half(cur_cell, r_cell)[ii] - find_f_half(l_cell, cur_cell)[ii]) / dx
        current_cell_later.U[ii] = cur_cell.U[ii] + dt * du
    return current_cell_later.U


def find_f_later(cur_cell_later):
    P = find_p(cur_cell_later)
    cur_cell_later.F[0] = cur_cell_later.U[1]
    cur_cell_later.F[1] = cur_cell_later.U[1] ** 2 / cur_cell_later.U[0] + P
    cur_cell_later.F[2] = (cur_cell_later.U[2] + P) * cur_cell_later.U[1] / cur_cell_later.U[0]
    # print(P)
    return cur_cell_later.F


for t in range(nt-1):
    for i, cell in enumerate(grid_cur[t][1:nx-1]):
        left_cell = grid_cur[t][i-1]
        right_cell = grid_cur[t][i+1]
        current_cell = grid_cur[t][i]
        left_cell_later = grid_cur[t+1][i - 1]
        right_cell_later = grid_cur[t+1][i + 1]
        current_cell_later = grid_cur[t+1][i]
        delta_x = 1
        delta_t = 0.1

        current_cell_later.U = find_u_later(current_cell, right_cell, left_cell, delta_x, delta_t)
        current_cell_later.T = find_f_later(current_cell)

        # print("alpha plus:", cell.alpha_plus)
        # print("alpha minus:", cell.alpha_minus)


def plot_grid(grid_to_plot, nt_cur, nx_cur):
    rho_map_cur = np.zeros((nt_cur, nx_cur))
    for jj in range(nt_cur):
        for ii in range(nx_cur):
            rho_map_cur[jj][ii] = grid_to_plot[jj][ii].U[0]
    plt.pcolor(rho_map_cur)
    plt.show()
    print(rho_map_cur)

plot_grid(grid_cur, nt, nx)
# plt.show()