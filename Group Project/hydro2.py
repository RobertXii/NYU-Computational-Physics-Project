import numpy as np
import matplotlib.pyplot as plt

#default value initialization
nx = 300
# ny = 1
v0 = 0.0 # initial velocity
delta_x = 1
delta_t = 0.1
gamma = 1.4
theta = 1
U = np.zeros((nx, 3))
c_int = np.zeros((nx, 3))
c_half_L = np.zeros((nx-2, 3))
c_half_R = np.zeros((nx-2, 3))
F = np.zeros((nx, 3))
F_half = np.zeros((nx, 3))
U_der = np.zeros((nx, 3))
UL = np.zeros((nx-2, 3))
UR = np.zeros((nx-2, 3))
FL = np.zeros((nx-2, 3))
FR = np.zeros((nx-2, 3))


def initialize():
    global U, nx, v0, delta_t, delta_x, gamma, theta, c_half_L, c_half_R, c_int, F, F_half, U_der, UL, UR, FL, FR
    #initialization
    nx = 300
    # ny = 1
    v0 = 0.0 # initial velocity
    delta_x = 1
    delta_t = 0.1
    gamma = 1.4
    theta = 1
    U = np.zeros((nx, 3))
    c_int = np.zeros((nx, 3))
    c_half_L = np.zeros((nx-2, 3))
    c_half_R = np.zeros((nx-2, 3))
    F = np.zeros((nx, 3))
    F_half = np.zeros((nx, 3))
    U_der = np.zeros((nx, 3))
    UL = np.zeros((nx-2, 3))
    UR = np.zeros((nx-2, 3))
    FL = np.zeros((nx-2, 3))
    FR = np.zeros((nx-2, 3))
    # initialization of U
    for i in range(nx):
        if i < nx / 2:
            rho_L = 1
            P_L = 0.8
            U[i, 0:3] = np.array([rho_L, rho_L*v0, P_L])
        else:
            rho_R = 0.1
            P_R = 0.1
            U[i, 0:3] = np.array([rho_R, rho_R*v0, P_R])

def sign(n):
    sign_matrix = np.zeros((nx - 2, 3))
    non_zero_mask = n != 0
    sign_matrix[non_zero_mask] = n[non_zero_mask] / abs(n[non_zero_mask])
    # print(sign_matrix)
    return sign_matrix

def minmod(x,y,z):
    minmod_matrix = np.zeros((nx - 2, 3))
    min_array = np.minimum(x, np.minimum(y, z))
    xx = sign(x)
    yy = sign(y)
    zz = sign(z)
    for i in range (nx-2):
        for j in range (3):
            minmod_matrix[i,j] = 0.25 * np.sqrt(np.square(xx[i,j] + yy[i,j]))*(xx[i,j] + zz[i,j]) * min_array[i,j]
    return minmod_matrix

def differences(list):
    return np.subtract(list[:][1:], list[:][:-1])

def differences2(list):
    return np.subtract(list[:][2:], list[:][:-2])

def find_c_half(U):
    global nx, v0, delta_t, delta_x, gamma, theta, c_half_L, c_half_R, c_int, F, F_half, U_der, UL, UR, FL, FR
    c_int[:, 0] = U[:, 0]
    c_int[:, 1] = (gamma-1)*U[:, 0]*(U[:, 2]/U[:, 0]-0.5*(U[:, 1]/U[:, 0])**2)
    c_int[:, 2] = U[:, 1]/U[:, 0]
    diff = differences(c_int)
    diff2 = differences2(c_int)
    c_half_L[:, :] = c_int[:-2, :] + 0.5 * minmod(theta*diff[:-1,:], 0.5*diff2, theta*diff[1:,:])
    c_half_R[:, :] = c_int[1:-1, :] + 0.5 * minmod(theta*diff[:-1,:], 0.5*diff2, theta*diff[1:,:])
    return c_half_L, c_half_R

def c_to_ULR_FLR(cL,cR):
    global U, nx, v0, delta_t, delta_x, gamma, theta, c_half_L, c_half_R, c_int, F, F_half, U_der, UL, UR, FL, FR
    UL[:, 0] = cL[:, 0]
    UL[:, 1] = cL[:, 0] * cL[:, 2]
    UL[:, 2] = cL[:, 1]/(gamma-1)+0.5*cL[:, 0]*cL[:, 2]**2
    FL[:, 0] = UL[:, 1]
    FL[:, 1] = cL[:, 0] * cL[:, 2] ** 2 + cL[:, 1]
    FL[:, 2] = (UL[:, 2]+cL[:, 1])*cL[:, 2]
    UR[:, 0] = cR[:, 0]
    UR[:, 1] = cR[:, 0] * cR[:, 2]
    UR[:, 2] = cR[:, 1]/(gamma-1)+0.5*cR[:, 0]*cR[:, 2]**2
    FR[:, 0] = UR[:, 1]
    FR[:, 1] = cR[:, 0] * cR[:, 2] ** 2 + cR[:, 1]
    FR[:, 2] = (UR[:, 2]+cR[:, 1])*cR[:, 2]
    return UL,UR,FL,FR
#
# aa, bb = find_c_half(U)
# c_to_ULR_FLR(aa, bb)

def find_f_half(UL,UR,FL,FR):
    global U, nx, v0, delta_t, delta_x, gamma, theta, c_half_L, c_half_R, c_int, F, F_half, U_der
    vL = UL[:, 1] / UL[:, 0]
    vR = UR[:, 1] / UR[:, 0]
    # print(vL)
    PL = (gamma-1)*UL[:,0]*(UL[:,2]/UL[:,0]-0.5*vL**2)
    PR = (gamma - 1) * UR[:, 0] * (UR[:, 2] / UR[:, 0] - 0.5 * vR ** 2)
    c_s_L = np.sqrt(PL*gamma/UL[:,0])
    c_s_R = np.sqrt(PR*gamma/UR[:,0])
    lambda_p_L = vL + c_s_L
    lambda_p_R = vR + c_s_R
    lambda_m_L = vL - c_s_L
    lambda_m_R = vR - c_s_R
    # print(lambda_m[1,0]*U[1,0,1])
    alpha = np.zeros((nx-2,6)) #first three rows alpha plus, second three alpha minus
    for i in range(nx-2):
        for k in range(3):
            alpha[i,k] = max(0, lambda_p_L[i], lambda_p_R[i])
            alpha[i,k+3] = max(0, -lambda_m_L[i], -lambda_m_R[i])
            # print(alpha.max())
    F_half = (alpha[:,0:3]*FL[:,:]+alpha[:,3:6]*FR[:,:]-alpha[:,0:3]*\
              alpha[:,3:6]*(UR[:,:]-UL[:,:]))/(alpha[:,0:3]+alpha[:,3:6])
    return F_half

def find_u_der(F_half, delta_x):
    global U, nx, v0, delta_t, gamma, theta, c_half_L, c_half_R, c_int, F, U_der, UL, UR, FL, FR

    U_der = -(F_half[1:,:]-F_half[:-1,:])/delta_x
    U_der = np.append(U_der,[np.array([0,0,0])], axis=0)
    U_der = np.vstack([np.array([0,0,0]),U_der,np.array([0,0,0])])
    # print(U_der)
    return U_der

def find_u(U, U_der,delta_t, ii):
    # print(U)
    # U_histor[i] = U[:nx-2,:,:]
    if(ii % 50 == 0):
        # np.save("data-part2/array"+str(ii)+".npy", U[:,1]/U[:,0])#velocity
        np.save("data-part2/array"+str(ii)+".npy", U[:,0])#density
        # np.save("data-part2/array"+str(ii)+".npy", U[:,0]*(U[:,2]/U[:,0]-0.5*(U[:,1]/U[:,0])**2)) # Pressure

    U = U + delta_t*U_der
    # print(U)
    return U


def evolve(i):
    global U, nx, v0, delta_t, gamma, theta, c_half_L, c_half_R, c_int, F, U_der, UL, UR, FL, FR

    cL, cR = find_c_half(U)
    UL, UR, FL, FR = c_to_ULR_FLR(cL,cR)
    F_half = find_f_half(UL, UR, FL, FR)
    U_der = find_u_der(F_half,delta_x)
    U1 = find_u(U,U_der,delta_t/3, ii=i)

    cL, cR = find_c_half(U)
    UL, UR, FL, FR = c_to_ULR_FLR(cL, cR)
    F_half2 = find_f_half(UL, UR, FL, FR)
    U_der2 = find_u_der(F_half2, delta_x)
    U2 = 0.75*U+0.25*U1+0.25*delta_t/3*U_der2

    cL, cR = find_c_half(U)
    UL, UR, FL, FR = c_to_ULR_FLR(cL, cR)
    F_half3 = find_f_half(UL, UR, FL, FR)
    U_der3 = find_u_der(F_half3,delta_x)
    U = 1/3*U+2/3*U2+2/3*delta_t/3*U_der3
    # if i % 50 == 0:
    #     plt.plot(range(nx), U[:,1]/U[:,0])
    # if i == 499:
    #     plt.show()

