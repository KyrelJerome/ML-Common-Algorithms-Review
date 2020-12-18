def trainLR(X,T,N,lrate):
  L,M = np.shape(X)
  # initialize weights
  w = init(M)
  w0 = 0
  # reshape vectors
  T = np.reshape(T,[L,1]) # column vector
  w = np.reshape(w,[1,M]) # row vector
  # perform N weight updates
  for n in range(N):
    # compute output
    Z = np.matmul(X,w.T) + w0 # column vector
    Y = 1/(1+np.exp(-Z)) # column vector
    # compute gradients
    gw = np.matmul((Y-T).T,X) # row vector. gradient wrt w
    gw0 = np.sum(Y-T) # real number. gradient wrt w0
    # update weights
    w -= lrate*gw # row vector
    w0 -= lrate*gw0 # real number
  return w,w0
