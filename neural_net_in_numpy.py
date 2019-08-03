
# ------------------------------------------------------------
# Single hidden layer neural network using numpy only
# done in 21h from 9days, gradient check/backprop took most of the time
# things to try: compare smaller parts of code to minimal_net.ipynb
# things I don't super get: affine broadcasting b1, softmax derivative
# ------------------------------------------------------------

import numpy as np
from numpy.random import RandomState
prng = RandomState(555)

def normalize(x, m, s): 
    return (x-m)/s

# Activation functions
def relu(x): 
    return np.clip(x, 0, np.finfo(x.dtype).max) # relu

def softmax(x):
    if len(x.shape) == 1:
        x = x.reshape((-1,1))
    return np.exp(x)/np.sum(np.exp(x), axis = 1).reshape((-1,1)) # softmax

# Inputs
X = prng.rand(50,18)
y = prng.randint(0,10,size=(50,))
classes = np.sort(np.array([str(i) for i in range(0,10)]))
num_nodes = 5 # nodes in hidden layer
reg = 1e-3 # regularization strength
step_size = 1 # for gradient descent
# Split train valid
vl = round(X.shape[0]*0.2)
vlid = prng.choice(range(0, X.shape[0]), size=(vl,), replace = False)
trid = np.array(list(set(range(0, X.shape[0])) - set(vlid)))
X_train, y_train, X_valid, y_valid = X[trid,:], y[trid,], X[vlid,:], y[vlid,]
y_true = np.zeros(shape = (num_obs,num_classes))
for i in range(0,num_classes):
    y_true[i,y_train[i]] = 1
# Derived
num_classes = len(classes)
num_obs = X_train.shape[0] # observations
num_features = X_train.shape[1] # features

# Normalize matrix values to -1,1 for faster convergence
train_mean,train_std = X_train.mean(),X_train.std()
X_train = normalize(X_train, train_mean, train_std) # m x n
# NB: Use training mean, not validation mean for validation set
X_valid = normalize(X_valid, train_mean, train_std) # 40x18 x 18x5 = 40x5, + 40x1 broadcasted

# Initialize weights
W1 = abs(prng.randn(num_nodes,num_features)) # weight matrix 1
W2 = abs(prng.randn(num_classes,num_nodes)) # weight matrix 2
B1 = np.ones((num_nodes,1)) # bias term for layer 1
B2 = np.ones((num_classes,1)) # bias term for layer 2

def forward_pass(x, y, w1, b1, w2, b2):
    """Perform forward pass for relu-softmax neural net"""
    # Input layer to hidden layer
    z1 = w1@x.T + b1 # nd x m
    a1 = relu(z1.T) # m x nd
    # Hidden layer to output layer
    z2 = w2@a1.T + b2 # c x m
    a2 = softmax(z2.T) # m x c
    # Calculate loss
    if len(y.shape) == 0:
        num_obs = 1
    else: num_obs = y.shape[0]
    j = -np.sum(y * np.log(a2)) / num_obs # data_loss
    j += 0.5*reg*np.sum(w2*w2) + 0.5*reg*np.sum(w1*w1) # reg loss
    return j, a2, a1

J, A2, A1 = forward_pass(X_train, y_true, W1, B1, W2, B2)

def backprop(x, y, a2, a1, w2, w1):
    """Calculate gradient for relu-softmax neural net"""
    # derivative of softmax: http://cs231n.github.io/neural-networks-case-study/#grad
    dscores = a2.copy() # m x c
    dscores -= y # dscores[range(m),y_train] -= 1
    if len(y.shape) == 0:
        num_obs = 1
    else: num_obs = y.shape[0]
    dscores /= num_obs
    dw2 = dscores.T@a1
    dw2 += reg*w2 # don't forget the regularization gradient
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # derivative of relu
    dhidden = dscores@w2 # m x nd
    dhidden[a1<=0] = 0
    dw1 = dhidden.T@x
    dw1 += reg*w1
    db1 = np.sum(dhidden, axis=0, keepdims=True)
    return dw1, db1, dw2, db2

dW1, dB1, dW2, dB2 = backprop(X_train, y_true, A2, A1, W2, W1)
# update parameters
W1_update = W1 - step_size*dW1
B1_update = B1 - step_size*dB1
W2_update = W2 - step_size*dW2
B2_update = B2 - step_size*dB2

# Gradient checking: https://www.coursera.org/learn/machine-learning/supplement/fqeMw/gradient-checking
epsilon = 1e-4
idx = 25 # prng.randint(0, X_train.shape[0], size=(1,))[0] # 26
pidx = (4,1)
# Sample 1 observation
x_samp = X_train[idx,:].reshape((1,-1)) # 1 row, n features
y_samp = y_true[idx,:]
# Analytical Gradient
J, A2, A1 = forward_pass(x_samp, y_samp, W1, B1, W2, B2)
dW1, dB1, dW2, dB2 = backprop(x_samp, y_samp, A2, A1, W2, W1)
# Numerical Gradient: +epsilon
W1_new = W1.copy()
W1_new[pidx] += epsilon # add small change to first first parameter
Jplus, A2, A1 = forward_pass(x_samp, y_samp, W1_new, B1, W2, B2)
# Numerical Gradient: -epsilon
W1_new = W1.copy()
W1_new[pidx] -= epsilon
Jminus, A2, A1 = forward_pass(x_samp, y_samp, W1_new, B1, W2, B2)

grc = (Jplus - Jminus)/(2*epsilon)
grc - dW1[pidx] < 1e-3 # compare gradient at W1

# # Compare against sklearn
# from sklearn.neural_network import MLPClassifier as mlp
# from sklearn.model_selection import train_test_split
# X_train, y_train, X_valid, y_valid = train_valid_split(X, y, valid_size = 0.2, random_state = 555)
# X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
# model = mlp(hidden_layer_sizes = (num_nodes,), activation = 'relu', solver = 'adam')
# model.fit(X_train,y_train)
# model.predict(X_valid)
