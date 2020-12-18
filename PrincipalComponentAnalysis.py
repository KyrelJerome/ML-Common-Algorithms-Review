def myPCA(X,K):
  # compute mean and covariance
  mu = np.mean(X,axis=0) # shape = [M]
  Xc = X - mu # centered data. shape = [N,M]
  cov = np.matmul(Xc.T,Xc)/len(X) # shape = [M,M]
  # eigenvectors
  U = la.eigh(cov)[1] # shape(U) = [M,M]
  U = U[:,-K:] # shape = [M,K]
  # reduce dimensionality
  Xr = np.matmul(Xc,U) # shape = [N,K]
  # reconstruct
  Xt = np.matmul(Xr,U.T) # shape = [N,M]
  return Xt + mu
