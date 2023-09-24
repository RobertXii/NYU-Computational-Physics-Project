import numpy as np
import matplotlib.pyplot as plt

N = 1000
tau = 3.053*60
mu = np.log(2)/tau

z = np.random.rand(N)

t = -1/mu*np.log(1-z)
t = np.sort(t)
decayed = np.arange(1,N+1)
survived = N - decayed

plt.plot(t, survived, label = 'Tl')
plt.plot(t, decayed, label = 'Pb')
#plt.plot(t,Pb, label = 'Pb')
plt.xlabel ("Time")
plt.ylabel("Number of atoms")
plt.legend()
plt.show()