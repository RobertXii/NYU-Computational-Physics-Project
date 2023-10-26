from math import sqrt

class Cell:

    #constants
    gamma = 1.4

    #initial conditions
    rho = 0
    v = 0
    e = 0
    E = 0
    P = 0
    lambda_plus = 0
    lambda_minus = 0

    #variables
    alpha_plus = [0, 0, 0]
    alpha_minus = [0, 0, 0]

    def __init__(self, rho, v, e) -> None:
        self.rho = rho
        self.v = v
        self.e = e
        self.P = (self.gamma-1)*self.rho*self.e
        self.E = rho*e + 1/2*rho*v**2.0
        self.U = [rho, rho*v, self.E]
        self.F = [rho*v,
                  rho*(v**2.0+(self.gamma - 1)*e),
                  rho*v*(1/2*v**2+self.gamma*e)]
        print(self.F)
        self.lambda_plus = v + sqrt((self.gamma*(self.gamma-1)*self.rho*self.e)/self.rho)
        self.lambda_minus = v - sqrt((self.gamma*(self.gamma-1)*self.rho*self.e)/self.rho)
        # print("v:", v)
        # print("lambda_plus:", self.lambda_plus)
        # print("lambda_minus:", self.lambda_minus)

    U = [0, 0, 0]
    F = [0, 0, 0]
    F_half = [0, 0, 0]
