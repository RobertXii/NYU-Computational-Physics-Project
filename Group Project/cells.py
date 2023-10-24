from math import sqrt

class Cell:

    #constants
    gamma = 1.4

    #initial conditions
    rho = 0.0
    v = 0.0
    E = 0.0
    P = 0.0
    lambda_plus = 0.0
    lambda_minus = 0.0
    F_half = [0.0, 0.0, 0.0]

    #variables
    alpha_plus = [0.0, 0.0, 0.0]
    alpha_minus = [0.0, 0.0, 0.0]

    def __init__(self, rho, v, P) -> None:
        self.rho = rho
        self.v = v
        self.P = P
        self.U = [self.rho, self.rho*v, self.P/(self.rho*(self.gamma-1))+0.5*self.rho*self.v**2]
        self.F = [self.rho*self.v,
                  self.rho*(self.v**2.0+self.P),
                  self.v*(self.P/(self.rho*(self.gamma-1))+0.5*self.rho*self.v**2 + self.P)]
        self.lambda_plus = 0.0
        self.lambda_minus = 0.0
        self.F_half = [0.0, 0.0, 0.0]
        # print("v:", v)
        # print("lambda_plus:", self.lambda_plus)
        # print("lambda_minus:", self.lambda_minus)
