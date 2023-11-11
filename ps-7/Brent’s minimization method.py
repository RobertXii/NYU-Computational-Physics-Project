import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def quadratic(f=None, a=None, b=None, c=None):
    f_a, f_b, f_c = f(a), f(b), f(c)
    return b - ((b - a)**2 * (f_b - f_c) - (b -c)**2 * (f_b - f_a)) / 2*((b - a) * (f_b - f_c) - (b -c) * (f_b - f_a))

def brent(f=None, a=None, c=None, tol=1.e-10):
    # initialization for b
    ratio = (3. - np.sqrt(5)) / 2
    b = a + ratio * (c - a)

    #plot the function
    x = np.linspace(-0.8, 1.2, 1000)
    plt.plot(x, f(x))

    length, dif = np.abs(b - a), np.abs(b - a)
    n = 0

    #loop until difference smaller than tol
    while np.abs(dif) > tol:
        new_min = quadratic(f, a, b, c)

        if np.abs(new_min - b) < length and a < new_min < c:
            a, c = b, c if new_min < b else b
            length, dif = dif, b - new_min

            x = np.array([b, new_min])
            plt.plot(x, f(x))
            b = new_min
        else:
            if b - a > c - b:
                new_min = b
                b -= ratio * (b - a)
            else:
                new_min = b + ratio * (c - b)
            length, dif = dif, b - new_min

            x = np.array([b, new_min])
            plt.plot(x, f(x))
            f_b = f(b)
            f_x = f(new_min)

            if (f_b < f_x):
                c = new_min
            else:
                a = b
                b = new_min
        n += 1

    plt.title("Brent's 1D Minimization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    return b, f(b), n

def function(x=None):
    return np.power((x-0.3),2) * np.exp(x)

def main():
    min_brent, _, n_brent = brent(function, 0, 2)
    print(
        f"The minimum obtained using Brent's 1D minimization is {min_brent}. Number of iterations: {n_brent}. Difference from actual minimum: {np.abs(0.3 - min_brent)}")

    res = scipy.optimize.brent(function, brack=(0, 1), full_output=True)
    min_scipy, _, n_scipy, _ = res
    print(
        f"The minimum obtained using Scipy is {min_scipy}. Number of iterations: {n_scipy}. Difference from actual minimum: {np.abs(0.3 - min_scipy)}")

if __name__ == '__main__':
    main()