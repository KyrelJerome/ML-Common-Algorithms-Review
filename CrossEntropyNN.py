def ceNN(clf,X,T):
  # compute a one-hot encoding of T
  N = len(T)
  C = clf.n_outputs_ # number of output classes
  cList = range(C)
  cList = np.reshape(cList,[1,C])
  T = np.reshape(T,[N,1])
  onehot = (T==cList) # shape = [N,C]
  # compute CE1
  logP1 = clf.predict_log_proba(X) # shape = [N,C]
  CE1 = -np.sum(logP1*onehot)/N
  # forward propagation
  W1,W2 = clf.coefs_
  b1,b2 = clf.intercepts_
  Z1 = np.matmul(X,W1) + b1
  H1 = 1/(1+np.exp(-Z1))
  Z2 = np.matmul(H1,W2) + b2
  # compute CE2
  logP2 = log_softmax(Z2)
  CE2 = -np.sum(logP2*onehot)/N
  return CE1, CE2
