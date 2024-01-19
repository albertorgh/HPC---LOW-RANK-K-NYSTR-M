# Disable numpy threading to get useful timings
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Import Libraries
import numpy as np
from numpy.linalg import norm
import time
from scipy.linalg import hadamard
import math
import random
import csv

# Define an iterative function to compute the Walsh-Hadamard normalized matrix
def HadamardRicorsiva(m):
    q = int(math.log2(m))
    H = np.array([[1, 1], [1, -1]])
    for i in range(q-1):
        H = np.block([[H, H], [H, -H]])
    return H/np.sqrt(m)

# This code is related to the mnist data set and a couple (l,k)=(80,20)

# Load the matrix
A = []
with open('matrix4096M.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        A.append([float(value) for value in row])
A = np.array(A)
m = A.shape[0]

# Take the time
wt = time.time()

# Define the sketching size
l = 80
# Choose the rank-k to approximate and extract the data with respect to l
k = 20

# Compute the random left diagonal matrix
Dl = np.diag(np.random.choice([-1, 1], size=l))
# Extract rows from the Identity matrix
I = np.eye(m)
indexes = [random.randint(0,m-1) for _ in range(l)]
R0 = I[indexes,:]
# Define and normalize the Hadamard matrix
H0 = HadamardRicorsiva(m)
RH = R0 @ H0
# Compute the random right diagonal matrix
Dr = np.diag(np.random.choice([-1, 1], size=m))
# Compute the sketching matrix and transpose it for the matrix dimensions
Omega0 = np.sqrt(m / l) * Dl @ RH @ Dr
Omega=np.transpose(Omega0)

# For Gaussian computations
# Omega = np.random.randn(m, l)*(1/np.sqrt(l))

# Compute B and C
C = A @ Omega
B = np.transpose(Omega) @ C
# Compute the Singular value decomposition of B and the related Z factor
U, Sigma, Vt = np.linalg.svd(B)
sqrt_sigma = np.linalg.pinv(np.diag(Sigma))
Sigmasqrt = np.sqrt(sqrt_sigma)
Z = C @ np.transpose(Vt) @ Sigmasqrt
# Compute the QR factorization
Q, R = np.linalg.qr(Z)
# Compute the singular value decomposition of the triangular matrix
U2, Sigma2, Vt2 = np.linalg.svd(R)
#V2 = np.transpose(Vt2)
# Extract columns and rows from the SVD
U_k = U2[:, 0:k]
Sigma_k = np.diag(Sigma2)[0:k, 0:k]
# V_k = V2[:, 0:k] for the more efficient one
U_cap_k = Q @ U_k
# U_cap_k = Z@V_k@np.linalg.pinv(Sigma_k) the more efficient one

# Assembly of the Nyström matrix
A_Nyst = U_cap_k @ Sigma_k @ Sigma_k @ np.transpose(U_cap_k)

# Conclude and print the runtime
wt=time.time()-wt
print("wt: ", wt)

wt_opt = time.time()

# Compute the optimal approximation
U_opt, S_opt, Vt_opt = np.linalg.svd(A)
V_opt = np.transpose(Vt_opt)
U_optk = U_opt[:,0:k]
S_optk = np.diag(S_opt)[0:k,0:k]
V_optk = V_opt[:,0:k]
A_opt = U_optk @ S_optk @ np.transpose(V_optk)

wt_opt = time.time() - wt_opt
print("wt_opt: ", wt_opt)

# Compare the relative errors
print(norm(A-A_opt,ord='nuc')/(norm(A,ord='nuc')))
print(norm(A-A_Nyst,ord='nuc')/norm(A,ord='nuc'))

