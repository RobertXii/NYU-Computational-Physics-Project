import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

#(a)
for i in range(10):
    plt.plot(logwave, flux[i, :])
plt.ylabel('flux ($10^{−17}$ erg s$^{−1}$ cm$^{−2}$ A$^{-1}$)')
plt.xlabel('wavelength ($A$)')
plt.savefig('10galaxies.png')
# plt.show()

#(b)
flux_norm = flux/np.tile(np.sum(flux, axis = 1), (np.shape(flux)[1], 1)).T

plt.plot(np.sum(flux_norm, axis = 1))
plt.ylabel('Sum')
plt.xlabel('different galaxies')
plt.ylim(0,2)
# plt.show()

#plot normalized data
for i in range(10):
    plt.plot(logwave, flux_norm[i, :])
plt.ylabel('normalized flux')
plt.xlabel('wavelength ($A$)')
plt.savefig('normalized10galaxies.png')
# plt.show()

#(c)
flux_0 = flux_norm-np.tile(np.mean(flux_norm, axis=1), (np.shape(flux)[1], 1)).T
plt.plot(logwave, flux_0[0, :])
plt.plot(np.sum(flux_0, axis = 1))
for i in range(10):
    plt.plot(logwave, flux_0[i, :])
plt.ylabel('normalized flux mean at zero')
plt.xlabel('wavelength ($A$)')
plt.savefig('normalized10galaxies0mean.png')
# plt.show()

#(d) covariance matrix
start_time_covariance = time.time()
R = flux_0
C = R.T@R
# print(len(C), len(C[0]))
eigenval, eigenvec = np.linalg.eig(C)
# print(eigenvec)
sortid = np.argsort(eigenval)[::-1] # sort eigenvalue give indices
# print(sortid)
eigenval_sorted = eigenval[sortid]
eigenvec_sorted = eigenvec[:, sortid]

end_time_covariance = time.time()

print("Computation time using covariance: " + str(end_time_covariance-start_time_covariance) + "s")

cU, cS, cV = np.linalg.svd(C)
print("Condition Number of C is", np.max(cS)/np.min(cS))

# for i in range(5):
#     plt.plot(eigenvec[:, i], label = f'Label {i}')
# plt.legend()
# plt.ylabel('eigen vector of covariance matrix')
# plt.savefig('5 eigen vectors using covariance matrix.png')
# plt.show()


#(e) SVD method
start_time_SVD = time.time()
U, S, V = np.linalg.svd(R) # single value decompsition
SVD_eigvec = V.T
SVD_eigval = S**2
SVD_sortid = np.argsort(SVD_eigval)[::-1] # sort eigenvalue give indices
SVD_eigvec = SVD_eigvec[:,SVD_sortid]
SVD_eigval = SVD_eigval[SVD_sortid]

end_time_SVD = time.time()

print("Computation time using SVD: " + str(end_time_SVD-start_time_SVD) + "s")
print("Condition Number of R is", np.max(S)/np.min(S))
for i in range(5):
    plt.plot(SVD_eigvec[:, i], label = f'Label {i}')
plt.legend()
plt.ylabel('eigen vectors using SVD')
plt.savefig('5 eigenvectors using SVD.png')
# plt.show()

[plt.plot(SVD_eigvec[:,i], eigenvec_sorted[:,i], 'o')for i in np.arange(10)]
plt.xlabel('SVD eigenvectors')
plt.ylabel('covariance eigenvectors')
plt.savefig('SVD vs Covariance eigenvector.png')
# plt.show()

[plt.plot(SVD_eigval[i], eigenval_sorted[i], 'o')for i in np.arange(50)]
plt.xlabel('SVD eigenvalues')
plt.ylabel('covariance eigenvalues')
plt.savefig('SVD vs Covariance eigenvalues.png')
# plt.show()


#(g)PCA
Nc = 5
eigvec_new = eigenvec_sorted[:,:Nc]
reduced= np.dot(eigvec_new.T, R.T)
approx = np.dot(eigvec_new, reduced).T

plt.plot(logwave, approx[0, :], label = 'Nc = 5')
plt.plot(logwave, R[0,:], label = 'original data')
plt.ylabel('flux')
plt.xlabel('wavelength ($A$)')
plt.legend()
plt.savefig('original data vs approximate data.png')
# plt.show()


#(h)
reduced_T = reduced.T
c0 = reduced_T[:,0]
c1 = reduced_T[:,1]
c2 = reduced_T[:,2]
plt.plot(c0,c1, '.')
plt.xlabel('c0')
plt.ylabel('c1')
plt.savefig('c0 vs c1.png')
# plt.show()

plt.plot(c0,c2, '.')
plt.xlabel('c0')
plt.ylabel('c2')
plt.savefig('c0 vs c2.png')
# plt.show()


#(i)
def pca(Nc, R):
    eigvec_new = eigenvec_sorted[:,:Nc]
    reduced= np.dot(eigvec_new.T, R.T)
    approx = np.dot(eigvec_new, reduced).T
    return approx

residual = np.zeros((20,len(C)))
sum_residual = np.zeros(20)
# print(len(C))
for i in range(2,21):
    residual[i-1,:] = ((pca(i,R)[0,:] - R[0,:])/R[0,:])**2
    sum_residual[i-1]= sum(residual[i-1,:])
#     print(i)
# print(sum_residual)

plt.plot(np.arange(2, 22),sum_residual, '.')
plt.xlabel('Nc')
plt.ylabel('residual')
plt.savefig('Nc vs residual.png')
# plt.show()




