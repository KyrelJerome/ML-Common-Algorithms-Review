def myGDA(Xtrain0,Xtrain1,Xtest):
# train the GDA classifier by fitting a mv Gaussian to each training set
  Mu0,Sigma0 = fit_mvn(Xtrain0)
  Mu1,Sigma1 = fit_mvn(Xtrain1)
  N0 = np.shape(Xtrain0)[0]
  N1 = np.shape(Xtrain1)[0]
  Prior0 = N0/float(N0+N1)
  Prior1 = 1 - Prior0
  # use Bayes rue to make predictions for points in Xtest
  P0 = mvn_pdf(Xtest,Mu0,Sigma0) * Prior0 # posterior probabilities for class 0
  P1 = mvn_pdf(Xtest,Mu1,Sigma1) * Prior1 # posterior probabilities for class 1
  T = np.argmax([P0,P1],axis=0) # predictions
  return T
