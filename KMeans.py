def myKmeans(X,K,I):
  N,M = np.shape(X)
  # initialize the means
  Mu = initMu2(X,K)
  # alternating minimization
  Mu = np.reshape(Mu,[1,K,M])
  X = np.reshape(X,[N,1,M])
  scoreList = [] # initialize list of scores
  for i in range(I):
    # compute scores (objective function)
    Dsq = np.sum((X-Mu)**2,axis=2,keepdims=True) # squared distances from points to cluster centers.
    shape = [N,K,1]
    Dmin = np.min(Dsq,axis=1,keepdims=True) # minimum squared distances. shape = [N,1,1]
    score = np.sum(Dmin)
    scoreList.append(score)
    # update assignnments (assignment step)
    R = (Dsq==Dmin) # one-hot encoding of Assignments. shape = [N,K,1]
    R = R.astype(np.int32)
    # update cluster centers (refitting step)
    Rsum = np.sum(R,axis=0,keepdims=True) # number of points in each cluster. shape = [1,K,1]
    Xsum = np.sum(X*R,axis=0,keepdims=True) # sum of points in each cluster. shape = [1,K,M]
    Mu = Xsum/Rsum # new cluster centers. shape=[1,K,M]
  Mu = np.reshape(Mu,[K,M])
  R = np.reshape(R,[N,K])
  return Mu,R,scoreList
