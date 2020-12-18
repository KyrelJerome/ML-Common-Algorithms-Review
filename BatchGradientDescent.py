def batch_gd(J,K,N,lrate):
  rnd.seed(0)
  Ntrain,I = np.shape(Xtrain2) # input shape
  # initialise the weights and biases
  W = rnd.randn(I,J) # random initial weights for hidden layer 1
  V = rnd.randn(J,K) # random initial weights for hidden layer 2
  U = rnd.randn(K,1) # random initial weights for output layer
  w0 = np.zeros([1,J]) # initial bias term for hidden layer 1
  v0 = np.zeros([1,K]) # initial bias terms for hidden layer 2
  u0 = 0 # initial bias term for output layer

  X= Xtrain2
  T = Ttrain2
  lrate = lrate/float(Ntrain)
  for n in range(N):
    # compute and print test accuracy
    O = forward(Xtest2,U,V,W,u0,v0,w0)[0]
    accTest = accuracy(O,Ttest2)
    print('Epoch', n,' Test accuracy =',accTest)

    # forward pass using training data (compute hidden values and output)
    O,G,H = forward(X,U,V,W,u0,v0,w0)

    # backward pass using training data (compute gradients of C)
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

    # compute and print final accuracy and cross entropy
    O = forward(Xtest2,U,V,W,u0,v0,w0)[0]
    print('Test accuracy =',accuracy(O,Ttest2))
    print('Cross entropy =',cross_entropy(O,Ttest2))
