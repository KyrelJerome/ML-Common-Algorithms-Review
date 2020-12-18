def sigmoid(Z):
  return 1/(1+np.exp(-Z))
  
def softmax(Z):
  expZ = np.exp(Z)
  expZsum = np.sum(expZ,axis=1,keepdims=True)
  return expZ/expZsum
 
 def ReLU(Z):
  if Z > 0:
	  return Z
  else:
	  return 0
