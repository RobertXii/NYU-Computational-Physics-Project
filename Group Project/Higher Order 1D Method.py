import numpy as np
import matplotlib.pyplot as plt

#initialization
nx = 300
# ny = 1
nt = 700
v0 = 0.0 # initial velocity
delta_x = 1
delta_t = 0.1
gamma = 1.4
theta = 1
U = np.zeros((nx, 3))
c_int = np.zeros((nx, 3))
c_half = np.zeros((nx-2, 3))
F = np.zeros((nx, 3))
F_half = np.zeros((nx, 3))
U_der = np.zeros((nx, 3))

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
    return n/abs(n)

def minmod(x,y,z):
    return 0.25* abs(sign(x) + sign(y))(sign(x) + sign(z))* min(abs(x), abs(y), abs(z))

# def differences(list):
#     return np.subtract(list[:][1:], list[:][:-1])
#
# def differences2(list):
#     return np.subtract(list[:][2:], list[:][:-2])
#
# def find_c_half(U):
#     c_int[:, 0] = U[:, 0]
#     c_int[:, 1] = (gamma-1)*U[:, 0]*(U[:, 2]/U[:, 0]-0.5*(U[:, 1]/U[:, 0])**2)
#     c_int[:, 2] = U[:, 1]/U[:, 0]
#     diff = differences(c_int)
#     diff2 = differences(c_int)
#     c_half[:, :] = c_int[1:-1, :] + 0.5 * minmod(theta*diff[:-1,:], 0.5*diff2, theta*diff[:-1,:])

def find_f(U):
    v = U[:, 1] / U[:,  0]
    P = (gamma-1)*U[:,0]*(U[:,2]/U[:,0]-0.5*v**2)
    e = P/((gamma-1)*U[:,0])
    F[:,0] = U[:,1]
    F[:,1] = U[:,1]*v + P
    F[:,2] = (U[:,0]*(e+0.5*v**2)+P)*v
    return F

def find_f_half(U,F):
    v = U[:,1]/U[:,0]
    P = (gamma-1)*U[:,0]*(U[:,2]/U[:,0]-0.5*v**2)
    c_s = np.sqrt(P*gamma/U[:,0])
    lambda_p = v + c_s
    lambda_m = v - c_s
    # print(lambda_m[1,0]*U[1,0,1])
    alpha = np.zeros((nx,6)) #first three rows alpha plus, second three alpha minus
    for i in range(nx-1):
        for k in range(3):
            alpha[i,k] = max(0, lambda_p[i], lambda_p[i+1])
            alpha[i,k+3] = max(0, -lambda_m[i], -lambda_m[i+1])
            # print(alpha.max())
    # print(alpha)
    F_half = (alpha[:nx-1,0:3]*F[:nx-1,:]+alpha[:nx-1,3:6]*F[1:,:]-alpha[:nx-1,0:3]*\
              alpha[:nx-1,3:6]*(U[1:,:]-U[:nx-1,:]))/(alpha[:nx-1,0:3]+alpha[:nx-1,3:6])
    return F_half

def find_u_der(F_half, delta_x):
    U_der = -(F_half[1:,:]-F_half[:-1,:])/delta_x
    U_der = np.append(U_der,[np.array([0,0,0])], axis=0)
    U_der = np.vstack([np.array([0,0,0]),U_der])
    # print(U_der)
    return U_der

def find_u(U, U_der,delta_t):
    # print(U)
    # U_histor[i] = U[:nx-2,:,:]
    U[:,:]= U[:,:] + delta_t*U_der
    # print(U)
    return U

for i in range(nt):
    F = find_f(U)
    F_half = find_f_half(U,F)
    U_der = find_u_der(F_half,delta_x)
    U1 = find_u(U,U_der,delta_t)
    F_half2 = find_f_half(U1,F)
    U_der2 = find_u_der(F_half2,delta_x)
    U2 = 0.75*U+0.25*U1+0.25*delta_t*U_der2
    F_half3 = find_f_half(U2,F)
    U_der3 = find_u_der(F_half3,delta_x)
    U = 1/3*U+2/3*U2+2/3*delta_t*U_der3
    if i % 50 == 0:
        plt.plot(range(nx), U[:,0])

plt.title('position vs density')
plt.show()
