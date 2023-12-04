import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#initialization
n = 30
nx = n
ny = n
nt = 100
v_x0 = 0
v_y0 = 0 # initial velocity
delta_x = 1
delta_t = 0.1
gamma = 1.4
U = np.zeros((nx, ny, 4))
G = np.zeros((nx, ny, 4)) ## G
F = np.zeros((nx, ny, 4))
F_half = np.zeros((nx, ny, 4))
G_half = np.zeros((nx, ny, 4))
U_der = np.zeros((nx, ny, 4))

# initialization of U
for i in range(nx):
    if i < nx / 2:
        rho_L = 1
        P_L = 0.8
        U[i,:, 0:4] = np.array([rho_L, rho_L*v_x0, rho_L*v_y0, P_L])
    else:
        rho_R = 0.1
        P_R = 0.1
        U[i,:, 0:4] = np.array([rho_R, rho_R*v_x0,rho_R*v_y0, P_R])
# print(U)

def find_FG(U):      ## Find f_U and G_U
    v_x = U[:, :, 1] / U[:, :, 0]  ## v in x-direction
    v_y = U[:, :, 2] / U[:, :, 0]  ## v in y-direction
    P = (gamma-1)*U[:, :, 0]*(U[:, :, 3]/U[:, :, 0]-0.5*(v_x**2+v_y**2))
    e = P/((gamma-1)*U[:, :, 0])
    F[:, :, 0] = U[:, :, 1]
    F[:, :, 1] = U[:, :, 1]*v_x**2 + P
    F[:, :, 2] = U[:, :, 1]*v_y*v_x
    F[:, :, 3] = (U[:, :, 0]*(e+0.5*(v_x**2+v_y**2))+P)*v_x
    G[:, :, 0] = U[:, :, 2]
    G[:, :, 1] = U[:, :, 1]*v_y*v_x
    G[:, :, 2] = U[:, :, 2]*v_y**2 + P
    G[:, :, 3] = (U[:, :, 0]*(e+0.5*(v_x**2+v_y**2))+P)*v_y
    # print(F)
    return F,G
# F,G = find_FG(U)

def find_f_half(U,F):
    v_x = U[:, :, 1] / U[:, :, 0]
    v_y = U[:, :, 2] / U[:, :, 0]
    P = (gamma-1)*U[:, :, 0]*(U[:, :, 3]/U[:, :, 0]-0.5*(v_x**2+v_y**2))
    c_s = np.sqrt(P*gamma/U[:, :, 0])
    lambda_p = np.sqrt(v_x**2+v_y**2) + c_s
    lambda_m = np.sqrt(v_x**2+v_y**2) - c_s
    alpha = np.zeros((nx,ny,8)) #first three rows alpha plus, second three alpha minus
    for i in range(nx-1):
        for j in range(ny-1):
            for k in range(4):
                alpha[i,j,k] = max(0, lambda_p[i][j], lambda_p[i+1][j+1])
                alpha[i,j,k+4] = max(0, -lambda_m[i][j], -lambda_m[i+1][j+1])
    # print(alpha.max())
    F_half = (alpha[:nx-1,:ny-1,0:4]*F[:nx-1,:ny-1,:]+alpha[:nx-1,:ny-1,4:8]*F[1:,1:,:]-alpha[:nx-1,:ny-1,0:4]*\
              alpha[:nx-1,:ny-1,4:8]*(U[1:,1:,:]-U[:nx-1,:ny-1,:]))/(alpha[:nx-1,:ny-1,0:4]+alpha[:nx-1,:ny-1,4:8]) ## eq5
    G_half = (alpha[:nx-1,:ny-1,0:4]*G[:nx-1,:ny-1,:]+alpha[:nx-1,:ny-1,4:8]*G[1:,1:,:]-alpha[:nx-1,:ny-1,0:4]*\
              alpha[:nx-1,:ny-1,4:8]*(U[1:,1:,:]-U[:nx-1,:ny-1,:]))/(alpha[:nx-1,:ny-1,0:4]+alpha[:nx-1,:ny-1,4:8]) ## eq5
    # print(F_half)
    # print(G_half)
    return F_half, G_half
# F_half, G_half = find_f_half(U,F)


def find_u_der(F_half, G_half, delta_x):
    U_der = -(F_half[1:,1:,:] - F_half[:-1,:-1, :])/delta_x - (G_half[:-1,:-1, :]-G_half[:-1,:-1,:])/delta_x
    # U_der = np.append(U_der,[np.array([0,0,0,0])], axis=0) #4 coloums
    # print(U_der)
    return U_der
# U_der = find_u_der(F_half, G_half, delta_x)

def find_u(U, U_der,delta_t,ii):
    # print(U)
    # U_histor[i] = U[:nx-2,:,:]
    U[1:-1, 1:-1, :] = U[1:-1, 1:-1, :] + delta_t * U_der
    if ii % 50 == 0:
        x_values, y_values = np.meshgrid(np.arange(nx), np.arange(ny))
        z_values = U[:,:,0]
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_values, y_values, z_values, cmap='viridis')
    # print(U)
    return U

# find_u(U, U_der,delta_t,1)

for i in range(nt):
    F,G = find_FG(U)
    F_half, G_half = find_f_half(U,F)
    U_der = find_u_der(F_half,G_half,delta_x)
    U = find_u(U,U_der,delta_t,i)

plt.show()