def stochastic_gd(J,K,N,lrate,batch_size):
  rnd.seed(0)
  Ntrain,I = np.shape(Xtrain2) # input shape
  # initialise the weights and biases
  W = rnd.randn(I,J) # random initial weights for hidden layer 1
  V = rnd.randn(J,K) # random initial weights for hidden layer 2
  U = rnd.randn(K,1) # random initial weights for output layer
  w0 = np.zeros([1,J]) # initial bias term for hidden layer 1
  v0 = np.zeros([1,K]) # initial bias terms for hidden layer 2
  u0 = 0 # initial bias term for output layer

  Xtrain = Xtrain2
  Ttrain = Ttrain2
  lrate = lrate/batch_size
  # perform N epochs of stochastic gradient descent
  for n in range(N):
    # compute and print test accuracy
    Otest = forward(Xtest2,U,V,W,u0,v0,w0)[0]
    accTest = accuracy(Otest,Ttest2)
    print('Epoch', n,' Test accuracy =',accTest)

    # copy and shuffle the training data randomly
    Xtrain,Ttrain = utils.shuffle(Xtrain,Ttrain)
    # perform one epoch of stochastic gradient descent
     ptr1 = 0 # pointer to start of mini-batch
    while ptr1 < Ntrain:
      # process one mini-batch of data.

      # get the next mini-batch
      ptr2 = np.min([ptr1+batch_size,Ntrain]) # pointer to end of mini-batch
      X = Xtrain[ptr1:ptr2]
      T = Ttrain[ptr1:ptr2]
      ptr1 = ptr2

      # forward pass (compute hidden values and output)
      O,G,H = forward(X,U,V,W,u0,v0,w0)

      # backward pass (compute gradients of C)
      # output layer
      gGt = O - T # gradient wrt Gt
      gU = np.matmul(G.T,gGt) # gradient wrt U
      gu0 = np.sum(gGt) # gradient wrt u0
      # hidden layer 2
      gG = np.matmul(gGt,U.T) # gradient wrt G
      gHt = (1-G**2)*gG # gradient wrt Ht
      gV = np.matmul(H.T,gHt) # gradient wrt V
      gv0 = np.sum(gHt,axis=0) # gradient wrt v0
      # hidden layer 1
      gH = np.matmul(gHt,V.T) # gradient wrt H
      gXt = (1-H**2)*gH # gradient wrt Xt
      gW = np.matmul(X.T,gXt) # gradient wrt W
      gw0 = np.sum(gXt,axis=0) # gradient wrt w0

      # update weight matrices
      U -= lrate*gU
      V -= lrate*gV
      W -= lrate*gW
      # update bias vectors
      u0 -= lrate*gu0
      v0 -= lrate*gv0
      w0 -= lrate*gw0
