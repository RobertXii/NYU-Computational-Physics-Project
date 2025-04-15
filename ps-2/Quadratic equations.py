import math
import numpy as np
from decimal import Decimal, getcontext

#I assume that there will be two roots!!
def Quadratic1(a,b,c):
    x = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
    y = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return x,y

def Quadratic2(a,b,c):
    x = 2*c/(-b - math.sqrt(b**2 - 4*a*c))
    y = 2*c/(-b + math.sqrt(b ** 2 - 4 * a * c))
    return x,y

    getcontext().prec = 30

def accurate(a, b, c):
    x = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    y = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return x,y

solution1 = Quadratic1(0.001, 1000, 0.001)
solution2 = Quadratic2(0.001, 1000, 0.001)
solution3 = accurate(Decimal("0.001"), Decimal("1000"), Decimal("0.001"))


print(f"(a)the two solutions for this equation is {solution1}")
print(f"(b)the two solutions for this equation is {solution2}")
print(f"(c)the two solutions for this equation is {solution3}")



