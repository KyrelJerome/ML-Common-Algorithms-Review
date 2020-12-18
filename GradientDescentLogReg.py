def gd_logreg(lrate):
  rnd.seed(3)
  # extend the data matrices
  EXtrain = extend(Xtrain)
  EXtest = extend(Xtest)
  Ntrain,M = np.shape(EXtrain)
  # initialization phase
  W = rnd.randn(M)/1000 # initialize the weight vector randomly
  Ztrain = EXtrain @ W # initial value of the linear function on the training data
  # initialize lists for recording error measures
  CEtrainList = [] # training cross entropy
  CEtestList = [] # test cross entropy
  accTrainList = [] # training accuracy
  accTestList = [] # test accuracy
  # initialize loop parameters
  CEchange = np.inf # change in cross entropy between iterations
  epsilon = 10**(-10) # threshold for loop termination
  I = 0 # number of iterations so far

  # perform gradient descent
  # while I < 200:
  while CEchange > epsilon:
    # perform one step of gradient descent
    Ytrain = logistic(Ztrain)
    gradW = EXtrain.T @ (Ytrain-Ttrain)/Ntrain # gradient of loss function wrt W
    W = W - lrate*gradW # update the weight vector
    # evaluate the linear function on the training and test data
    Ztrain = EXtrain @ W
    Ztest = EXtest @ W
    # compute error measures
    CEtrain = cross_entropy(Ztrain,Ttrain)
    CEtest = cross_entropy(Ztest,Ttest)
    accTrain = accuracy(Ztrain,Ttrain)
    accTest = accuracy(Ztest,Ttest)
     # record error measures
    CEtrainList.append(CEtrain)
    CEtestList.append(CEtest)
    accTrainList.append(accTrain)
    accTestList.append(accTest)
    # change in training error
    if I > 0:
      CEchange = CEtrainList[-2] - CEtrain
    I = I+1
