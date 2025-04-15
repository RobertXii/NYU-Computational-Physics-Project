from random import random
from numpy import arange
import matplotlib.pyplot as plt

# Constants
NBi213 = 10000
NTi = 0
NPb = 0
NBi209 = 0
tauBi213 = 46 * 60
tauPb = 3.3 * 60
tauTi = 2.2 * 60
h = 1.0
pBi213 = 1 - 2**(-h/tauBi213)
pTi = 1 - 2**(-h/tauTi)
pPb = 1 - 2**(-h/tauPb)
pBi213toTi = 0.0209
tmax = 20000

# Lists of plot points
tpoints =arange(0.0,tmax,h)
Bi213points = []
Tipoints = []
Pbpoints = []
Bi209points = []

# Main loop
for t in tpoints:
    Bi213points.append(NBi213)
    Tipoints.append(NTi)
    Pbpoints.append(NPb)
    Bi209points.append(NBi209)

    #Bi213 to Ti or Pb
    decayBi213toTi = 0
    decayBi213toPb = 0
    for i in range(NBi213):
        if random()<pBi213:
            NBi213 -= 1
            if random()<pBi213toTi:
                decayBi213toTi += 1
            else:
                decayBi213toPb += 1
    NTi += decayBi213toTi
    NPb += decayBi213toPb

    #Ti to Pb
    decayTi = 0
    for i in range(NTi):
        if random()<pTi:
            decayTi += 1
    NTi -= decayTi
    NPb += decayTi
    # Pb to Bi209
    decayPb = 0
    for i in range(NPb):
        if random()<pPb:
            decayPb += 1
    NPb -= decayPb
    NBi209 += decayPb

#show image
plt.plot(tpoints,Bi213points, label = 'Bi213')
plt.plot(tpoints,Tipoints,label = 'Ti')
plt.plot(tpoints,Pbpoints,label = 'Pb')
plt.plot(tpoints,Bi209points, label = 'Bi209')
plt.xlabel ("Time/s")
plt.ylabel("Number of atoms")
# Add a legend
plt.legend()
plt.show()
