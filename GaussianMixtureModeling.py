def myGMM(X,K,I):
  N,M = np.shape(X)
  # initialize the means and mixing coefficients
  Mu = initMu2(X,K)
  Pi = np.ones([K])/K # each mixing coefficient gets the same value, 1/K
  # # alternate method of initializing Pi,
  # # where each mixing coefficient gets a uniform random value.
  # Pi = rnd.rand(K)
  # Pi = Pi/np.sum(Pi)

  # alternating minimization
  X = np.reshape(X,[N,1,M])
  Mu = np.reshape(Mu,[1,K,M])
  Pi = np.reshape(Pi,[1,K,1])
  scoreList = [] # initialize list of scores
  C = (2*np.pi)**(M/2.0) # normalizing constant for Gaussian probabilities
  for i in range(I):
    # compute scores (mean log-likelihood)
    Dsq = np.sum((X-Mu)**2,axis=2,keepdims=True) # squared distances from points to cluster centers.
    shape = [N,K,1]
    P = Pi*np.exp(-Dsq/2)/C # joint probabilities. shape = [N,K,1]
    Psum = np.sum(P,axis=1,keepdims=True) # Likelihood. shape = [N,1,1]
    score = np.mean(np.log(Psum)) # mean log-likelihood
    scoreList.append(score)
    # update responsibilities (E step)
    R = P/Psum # responsibilities. shape = [N,K,1]
    # update Mu and Pi (M step)
    Rsum = np.sum(R,axis=0,keepdims=True) # number of points in each cluster. shape = (1,K,1)
    Xsum = np.sum(X*R,axis=0,keepdims=True) # sum of points in each cluster. shape = [1,K,M]
    Mu = Xsum/Rsum # new cluster centers. shape=[1,K,M]
    Pi = Rsum/N # shape = [1,K,1]
  Mu = np.reshape(Mu,[K,M])
  Pi = np.reshape(Pi,[K])
  R = np.reshape(R,[N,K])
  return Mu,Pi,R,scoreList
