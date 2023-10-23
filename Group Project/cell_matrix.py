import matplotlib.pyplot as plt
import numpy as np

#initalization
gamma = 1.4
v_0 = 0.1
rho_L = 10
rho_R = 1
P_L = 8
P_R = 1
delta_t = 1
delta_x = 1

nx = 20 #number of cells
nt = 5 #total time steps

U = np.zeros((3*nt, nx)) #create U zeros matrix for all time steps
F = np.zeros((3*nt, nx)) #create F zeros matrix for all time steps
F_half = np.zeros((3*nt, nx))
alpha = np.zeros((6*nt, nx)) #create matrix for alpha, first three rows plus, second three rows minus

for i in range(nx):
    if i < nx/2:
        U[0, i] = rho_L #initialize rho
        U[1, i] = rho_L * v_0
        U[2, i] = P_L / (gamma - 1)
        F[0, i] = rho_L * v_0
        F[1, i] = P_L + rho_L * v_0 ** 2
        F[2, i] = (P_L / (gamma - 1) + P_L) * v_0
    else:
        U[0, i] = rho_R
        U[1, i] = rho_R * v_0
        U[2, i] = P_R / (gamma - 1)
        F[0, i] = rho_R * v_0
        F[1, i] = P_R + rho_R * v_0 ** 2
        F[2, i] = (P_R / (gamma - 1) + P_R) * v_0

def find_alpha(t):  #find alpha plus minus at time t for all cells
    for i in range(nx-1):
        current_P = F[3 * t + 1, i] - F[3 * t, i]**2 / U[3 * t, i]
        current_v = U[3 * t + 1, i] / U[3 * t, i]
        current_rho = U[3 * t, i]
        next_P = F[3 * t + 1, i + 1] - F[3 * t, i + 1]**2 / U[3 * t, i + 1]
        next_v = U[3 * t + 1, i + 1] / U[3 * t, i + 1]
        next_rho = U[3 * t, i + 1]
        for j in range(3):
            alpha[6 * t + j, i] = max(0, (current_v + (gamma*current_P/current_rho) ** 0.5) * U[3 * t + j, i],
                                      (next_v+(gamma*next_P/next_rho) ** 0.5) * U[3 * t + j, i + 1])
            alpha[6 * t + 3 + j, i] = max(0, -(current_v - (gamma*current_P/current_rho) ** 0.5) * U[3 * t + j, i],
                                          -(next_v -(gamma*next_P/next_rho) ** 0.5) * U[3 * t + j, i + 1])

def find_F_half(t):
    for i in range(nx - 1):
        for j in range(3):
            F_half[3 * t + j, i] = (alpha[6 * t + j, i] * F[3 * t + j, i] + alpha[6 * t + j + 3, i] * F[3 * t + j, i+1]
                                    - alpha[6 * t + 3 + j, i] * alpha[6 * t + j, i] * (U[3 * t + j, i + 1] - U[3 * t + j, i]))\
                                   /(alpha[6 * t + j + 3, i] + alpha[6 * t + j, i])

def find_later_U(t, delta_t, delta_x):
    for i in range(nx - 1):
        for j in range(3):
            U[3 * (t + 1) + j, i] =  U[3 * t + j, i] - delta_t * ((F_half[3 * t + j, i]
                                                                         - F_half[3 * t + j, i - 1])/delta_x)
            #print((F_half[3 * t + (j - 1), i]- F_half[3 * t + (j - 1), i - 1]))

def find_later_F(t):
    for i in range(nx - 1):
        current_P = (gamma-1)*U[3 * (t + 1) - 1, i]*\
                                (U[3 * (t + 1)+1, i]/U[3 * (t + 1)-1, i]-0.5*(U[3 * (t + 1), i]/U[3 * (t + 1)-1, i])**2)
        F[3 * (t + 1), i] = U[3 * (t + 1) + 1, i]
        F[3 * (t + 1) + 1, i] = U[3 * (t + 1), i]**2 / U[3 * (t + 1)-1, i] + current_P
        F[3 * (t + 1) + 2, i] = (U[3 * (t + 1) + 1, i] + current_P) * U[3 * (t + 1) , i]/U[3 * (t + 1)-1, i]

for i in range(nt-1):
    find_alpha(i)
    find_F_half(i)
    find_later_U(i, delta_t, delta_x)
    find_later_F(i)
    #plt.plot(range(nx), U[3*i,:])

print(U)



plt.show()
