import numpy as np

def dense_full_rank_matrix(n):
  # Generate a square matrix of size n x n with random integer entries.
  # We keep generating matrices until we find one that is full rank.
  # For random integer matrices, the probability of being singular is very low,
  # especially for larger n.
  while True:
    # Generate random integer entries, for example, between 0 and 10.
    matrix = np.random.randint(-9,9, size=(n, n))
    if np.linalg.matrix_rank(matrix) == n:
      print("matrix is full rank")
      return matrix.astype(int)
    else:
      print("not full rank, regenerating...")

A = dense_full_rank_matrix(5)
print(A)

import scipy
from scipy import sparse

def generate_full_rank_sparse_integer_matrix(n, density, min, max):
    """
    Generates a random sparse matrix (m x n) that is likely to have full rank.
    It iteratively generates matrices until one with full rank is found.

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        density (float): Density of non-zero elements (0 to 1).

    Returns:
        A sparse matrix with full rank.
    """
    min_dim = n
    # A random matrix has full rank with high probability
    # The probability is higher with more non-zero elements (higher density)
    # or a wider value distribution.

    # Use a loop to ensure full rank, though it is highly probable on the first try
    while True:
        # Generate random sparse matrix using scipy.sparse.random
        # data_rvs uses a standard normal distribution for non-zero values
        # rvs = np.random.randn
        # M = sparse.random(n, n, density=density, data_rvs=rvs, format='csr')
        # M = M.toarray()
        # M = 50*M
        # M = M.astype(int)

        rng = np.random.default_rng()
        # Uniform distribution between min and max
        rvs = lambda size: rng.uniform(min, max, size=size)
        M = sparse.random(n, n, density=density, rng=rng, data_rvs=rvs, format='csr')
        M = M.toarray()
        M = M.astype(int)

        # Check the rank.
        # For very large matrices, use scipy.sparse.linalg.matrix_rank directly
        # as converting to a dense array might use too much memory.
        # For smaller matrices, you can convert to dense: rank = np.linalg.matrix_rank(M.toarray())
        rank = np.linalg.matrix_rank(M)

        if rank == min_dim:
            print(f"Generated matrix of shape {M.shape} with full rank: {rank}")
            return M
        else:
            print(f"Generated matrix of shape {M.shape} had rank {rank}, regenerating...")

# Maximal Bipartite matching.
# Used for finding the permutation that removes 0s on the diagonal
class GFG:
    def __init__(self,graph):

        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])

        '''An array to keep track of the
           applicants assigned to jobs.
           The value of matchR[i] is the
           applicant number assigned to job i,
           the value -1 indicates nobody is assigned.'''
        self.matchR = [-1] * self.jobs

    # A DFS based recursive function
    # that returns true if a matching
    # for vertex u is possible
    def bpm(self, u, matchR, seen):

        # Try every job one by one
        for v in range(self.jobs):

            # If applicant u is interested
            # in job v and v is not seen
            if self.graph[u][v] and seen[v] == False:

                # Mark v as visited
                seen[v] = True

                '''If job 'v' is not assigned to
                   an applicant OR previously assigned
                   applicant for job v (which is matchR[v])
                   has an alternate job available.
                   Since v is marked as visited in the
                   above line, matchR[v]  in the following
                   recursive call will not get job 'v' again'''
                if self.matchR[v] == -1 or self.bpm(self.matchR[v],
                                               self.matchR, seen):
                    self.matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):

            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs

            # Find if the applicant 'u' can get a job
            if self.bpm(i, self.matchR, seen):
                result += 1
        return result

################################################################################
# Find permutation for non-zeros on the diagonal of A using bipartite matching
def find_permutation(A):
  '''
  Finds a row permutation of A such that there are no
  zeros on the diagonal. Note, this is not a heuristic for
  minimizing fill-in for the factorization.
  A permutation vector is returned.
  '''
  n = A.shape[0]
  # Get the non-zero graph
  nzg = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      if A[i,j] != 0:
        nzg[i,j] = 1

  #Turn the graph into a bipartite matching
  g = GFG(nzg)
  if g.maxBPM() != n:
    raise ValueError("Matrix is not full rank")
  else:
    print("Matrix is full rank")
    P = g.matchR
    return P

################################################################################
# Dense REF triangular solve
def dense_REF_fsub(k, LU, a):
  '''
  Solves a system of the form LDx=a for x
  L is a dense integral lower triangular matrix of LU
  D = diag(rho_0*rho_1, rho_1*rho_2, ..., rho_n-1*rho_n)
  a is an integral vector
  '''
  n = LU.shape[0]
  x = a # Initialize x
  for j in range(1, n, 1):
    # Iterate over each row, but special for when i=0 to get LU[-1,-1]=0
    x[j] = LU[0,0]*x[j] - LU[j,0]*x[0] # IPGE
    for i in range(1, min(j,k)):
      x[j] = (LU[i,i]*x[j] - LU[j,i]*x[i])/LU[i-1,i-1] # IPGE
  return x

################################################################################
# Dense Left-looking LU factorization
def dense_LL_LU(P, A, Q):
  '''
  Return P, L, U for the system PA = LU
  '''
  n = A.shape[0]

  # Shared LU matrix
  LU = np.eye(n, dtype=int)
  # Solve for each column one-by-one
  for k in range(n):
    x = dense_REF_fsub(k, LU, A[P,Q[k]])
    LU[:,k] = x
  return LU

n = 6
A = generate_full_rank_sparse_integer_matrix(n, 0.7, -5, 5)
print(A)

P = find_permutation(A)
Q = list(range(n))

LU = dense_LL_LU(P, A, Q)
print("A=\n", A)
print("P=\n", P)
print("Q=\n", Q)
print("PAQ=\n", A[P,:][:,Q])
print("LU=\n", LU)

def verify_LU(P, A, Q, LU):
  '''
  Verify that PAQ = LDU
  Very inefficient programming
  '''
  n = A.shape[0]
  print("LU=\n", LU)

  L = np.zeros((n,n), dtype=int)
  for i in range(n):
    for j in range(i+1):
      L[i,j] = LU[i,j]
  print("L=\n", L)

  U = np.zeros((n,n), dtype=int)
  for j in range(n):
    for i in range(j+1):
      U[i,j] = LU[i,j]
  print("U=\n", U)

  D = np.zeros((n,n))
  D[0,0] = 1/LU[0,0]
  for i in range(1, n, 1):
    D[i,i] = 1/(LU[i-1,i-1]*LU[i,i])
  print("D=\n",D)

  print("PAQ=\n", A[P,:][:,Q])
  print("LDU=\n", np.round(L@D@U, 1))
  print("PAQ-LDU=\n", np.sum(np.sum((A[P,:][:,Q] - L@D@U) > 1e-10)))

def frame_to_LU(F):
  n = len(F)
  L = [[0 for _ in range(n)] for _ in range(n)]
  U = [[0 for _ in range(n)] for _ in range(n)]
  rhos = [0 for _ in range(n)]
  rhos.insert(0, 1)
  for i in range(n):
    for j in range(n):
      if i > j:
        # in L
        L[i][j] = F[i][j]
      elif i == j:
        # in D
        L[i][j] = F[i][j]
        U[i][j] = F[i][j]
        rhos[i+1] = F[i][j]
      else:
        # in U
        U[i][j] = F[i][j]
  return L, rhos, U

def sLTCpush(k, L, rhos, fP, fQ, Ftmp, rhosp, sL, sU, mu, h):
  # Backtracking
  print("Starting backtracking")
  if sU == 0:
    print("sU = 0, so no seed updates")
    for i in range(k+2, n):
      Ftmp[fP[i]][fQ[k]] = (rhosp[k]*L[i][k+1])/(rhos[k+1])
  else:
    print("sU != 0, may need some history updates")
    for i in range(k+2, n):
      # Need to do history updates to k-1 for when mu[i] is not 0
      if h[i] != k-1 and mu[i] != 0:
        mu[i] = (rhos[k+1]*mu[i])/rhos[h[i]+2]
        h[i] = k-1
      # Backtrack and place in Ftmp
      Ftmp[fP[i]][fQ[k]] = (rhosp[k]*L[i][k+1]+sU*mu[i])/(rhos[k+1])
  # RwSOP
  print("No RwSOP in L for column push")
  print("Re-initialize muL")
  if k < n-2:
    for i in range(n):
      mu[i] = -L[i][k+1]
      h[i] = k
    print("new muL should be:", mu)
  else:
    print("second to last push, skip RwSOP")

def sUTCpush(k, Ut, rhos, fP, fQ, Ftmp, rhosp, sL, sU, mu, h):
  # Backtracking
  print("No backtracking here")
  print("Copy down values from muU")
  for i in range(k+2, n):
    if h[i] != k-1 and mu[i] != 0:
      mu[i] = (rhos[k+1]*mu[i])/rhos[h[i]+2]
      h[i] = k-1
    Ftmp[fP[k]][fQ[i]] = mu[i]
  # RwSOP
  print("Starting RwSOP")
  if k < n-2:
    if sU == 0:
      # Just bring the next support up to k
      mu[k+2] = (rhos[k+2]*mu[k+2])/rhos[h[k+2]+2]
      h[k+2] = k
    else:
      if h[k+2] != k-1 and mu[k+2] != 0:
        mu[k+2] = (rhos[k+1]*mu[k+2])/rhos[h[k+2]+2]
        h[k+2] = k-1
      mu[k+2] = -(rhos[k+2]*mu[k+2]-sU*Ut[k+2][k+1])/rhos[k+1]
      h[k+2] = k
      for i in range(k+3, n):
        if h[i] != k-1:
          mu[i] = (rhos[k+1]*mu[i])/rhos[h[i]+2]
          h[i] = k-1
        print("calculating muU using")
        print(rhos[k+2], mu[i], sU, Ut[i][k+1], rhos[k+1])
        mu[i] = -(rhos[k+2]*mu[i]-sU*Ut[i][k+1])/rhos[k+1]
        h[i] = k
  else:
    print("second to last push, skip RwSOP")

def sLTDpush(ind, k, L, rhos, fP, fQ, Ftmp, rhosp, sL, sU, mu, h):
  # Backtracking
  print("Starting backtracking")
  if sU == 0:
    print("sU = 0, so no seed updates")
    for i in range(k+2, n):
      if ind == "L":
        Ftmp[fP[i]][fQ[k]] = (rhosp[k]*L[i][k+1])/(rhos[k+1])
      elif ind == "U":
        Ftmp[fQ[k]][fP[i]] = (rhosp[k]*L[i][k+1])/(rhos[k+1])
      else:
        raise ValueError("ind must be 'L' or 'U'")
  else:
    print("sU != 0, may need some history updates")
    for i in range(k+2, n):
      # Need to do history updates to k-1 for when mu[i] is not 0
      if h[i] < k-1 and mu[i] != 0:
        mu[i] = (rhos[k+1]*mu[i])/rhos[h[i]+2]
        h[i] = k-1
      # Backtrack and place in Ftmp
      if ind == "L":
        Ftmp[fP[i]][fQ[k]] = (rhosp[k]*L[i][k+1]+sU*mu[i])/(rhos[k+1])
      elif ind == "U":
        Ftmp[fQ[k]][fP[i]] = (rhosp[k]*L[i][k+1]+sU*mu[i])/(rhos[k+1])
      else:
        raise ValueError("ind must be 'L' or 'U'")
  # RwSOP
  print("Starting RwSOP")
  if k < n-2:
    if sL == 0:
      # Just bring the next support up to k
      mu[k+2] = (rhos[k+2]*mu[k+2])/rhos[h[k+2]+2]
      h[k+2] = k
    else:
      if h[k+2] != k-1 and mu[k+2] != 0:
        mu[k+2] = (rhos[k+1]*mu[k+2])/rhos[h[k+2]+2]
        h[k+2] = k-1
      mu[k+2] = (rhos[k+2]*mu[k+2]-sL*L[k+2][k+1])/rhos[k+1]
      h[k+2] = k
      for i in range(k+3, n):
        if h[i] != k-1:
          mu[i] = (rhos[k+1]*mu[i])/rhos[h[i]+2]
          h[i] = k-1
        mu[i] = (rhos[k+2]*mu[i]-sL*L[i][k+1])/rhos[k+1]
        h[i] = k
  else:
    print("second to last push, skip RwSOP")

def SDIPPS(A, K, PA, QA, L, U, rhos):
  # Transpose U (a dense upper triangular matrix into a lower triangular matrix)
  n = len(L)
  Ut = [list(row) for row in zip(*U)]
  # Initialize new pivots
  rhosp = [0 for _ in range(n)]
  rhosp.insert(0, 1)
  for i in range(K):
    rhosp[i+1] = rhos[i+1]
  print("rhosp initialized as \n", rhosp)
  # Initialize the temporary frame matrix
  # The first K-1 frames are filled
  Ftmp = [[0 for _ in range(n)] for _ in range(n)]
  for k in range(K):
    for i in range(k, n):
      Ftmp[i][k] = L[i][k]
      Ftmp[k][i] = U[k][i]
  print("Ftmp initialized as \n", np.array(Ftmp))
  fP = [_ for _ in range(n)]
  fQ = [_ for _ in range(n)]
  fPt = [_ for _ in range(n)]
  fQt = [_ for _ in range(n)]
  # Initialize the seed vectors and history
  muL = [L[_][K] for _ in range(n)]
  print("muL initialized as \n", muL)
  hL = [K-1 for _ in range(n)]
  muU = [Ut[_][K] for _ in range(n)]
  print("muU initialized as \n", muU)
  hU = [K-1 for _ in range(n)]
  for k in range(K, n-1):
    print("Running iteration k=", k)
    sL = muL[k+1]
    sU = muU[k+1]
    print("sL and sU are ", sL, sU)
    if sU != 0:
      print("\n\n Column Push \n\n")
      # Update all permutations
      # The PAQ permutations
      QA[k], QA[k+1] = QA[k+1], QA[k]
      # The fP and fQ permutations
      fQ[k], fQ[k+1] = fQ[k+1], fQ[k]
      print("New fP and fPt:")
      print(fP)
      print("New fQ and fQt:")
      print(fQ)
      # Fill in the 3 corner elements of frame k of F
      rhosp[k+1] = sU
      Ftmp[fP[k]][fQ[k]] = sU
      Ftmp[fP[k]][fQ[k+1]] = rhos[k+1]
      Ftmp[fP[k+1]][fQ[k]] = (rhosp[k]*rhos[k+2]+sU*sL)/rhos[k+1]
      sLTCpush(k, L, rhos, fP, fQ, Ftmp, rhosp, sL, sU, muL, hL)
      print("new muL\n", muL)
      print(hL)
      sUTCpush(k, Ut, rhos, fP, fQ, Ftmp, rhosp, sL, sU, muU, hU)
      print("new muU\n", muU)
      print(hU)
      print("Ftmp=\n", np.array(Ftmp, dtype=int))
      # Negate frames k+2:n
      rhos[k+2] = -rhos[k+2]
      for j in range(k+2, n):
        rhos[j+1] = -rhos[j+1]
        for i in range(j,n):
          L[i][j] = -L[i][j]
          Ut[i][j] = -Ut[i][j]
      F = np.array(Ftmp, dtype=int)
      F = F[fP, :][:, fQ]
      print("F=\n", F)
      LU = dense_LL_LU(P, A, Q)
      print("LU=\n", LU)
      print("rhos=\n", rhos)
      print("rhosp=\n", rhosp)
    elif rhosp[k]*rhos[k+2] + sL*sU != 0:
      print("\n\n Diagonal Push \n\n")
      # Update all permutations
      # The PAQ permutations
      PA[k], PA[k+1] = PA[k+1], PA[k]
      QA[k], QA[k+1] = QA[k+1], QA[k]
      # The fP and fQ permutations
      fP[k], fP[k+1] = fP[k+1], fP[k]
      fQ[k], fQ[k+1] = fQ[k+1], fQ[k]
      print("New fP and fPt:")
      print(fP)
      print("New fQ and fQt:")
      print(fQ)
      # Fill in the 3 corner elements of frame k of F
      rhosp[k+1] = (rhosp[k]*rhos[k+2] + sL*sU)/rhos[k+1]
      Ftmp[fP[k]][fQ[k]] = rhosp[k+1]
      Ftmp[fP[k]][fQ[k+1]] = sL
      Ftmp[fP[k+1]][fQ[k]] = sU
      sLTDpush("L", k, L, rhos, fP, fQ, Ftmp, rhosp, sL, sU, muL, hL)
      print("new muL\n", muL)
      print(hL)
      sLTDpush("U", k, Ut, rhos, fP, fQ, Ftmp, rhosp, sU, sL, muU, hU)
      print("new muU\n", muU)
      print(hU)
      print("Ftmp=\n", np.array(Ftmp, dtype=int))
      F = np.array(Ftmp, dtype=int)
      F = F[fP, :][:, fQ]
      print("F=\n", F)
      LU = dense_LL_LU(P, A, Q)
      print("LU=\n", LU)
    else:
      raise ValueError("No push available")
  Ftmp[fP[n-1]][fQ[n-1]] = rhos[n]
  print("Ftmp=\n", np.array(Ftmp, dtype=int))
  F = [[Ftmp[i][j] for j in fQ] for i in fP]
  return P, Q, F

n = 10
A = generate_full_rank_sparse_integer_matrix(n, 0.4, -9, 9)
print(A)

P = find_permutation(A)
Q = list(range(n))

LU = dense_LL_LU(P, A, Q)
print("A=\n", A)
print("P=\n", P)
print("Q=\n", Q)
print("PAQ=\n", A[P,:][:,Q])
print("LU=\n", LU)

F = LU.tolist()
L, rhos, U = frame_to_LU(F)
print("L=\n", np.array(L))
print("U=\n", np.array(U))
print("rhos=\n", rhos)

P, Q, F = SDIPPS(A, 0, P, Q, L, U, rhos)
print("F=\n", np.array(F, dtype=int))
LU = dense_LL_LU(P, A, Q)
print("LU=\n", LU)
abs_sum_diff = np.sum(np.abs(np.array(F) - np.array(LU)))
print("abs_sum_diff=", abs_sum_diff)
