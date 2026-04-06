import numpy as np

def sigmoid(Z): 
    return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))

def relu(Z): 
    return np.maximum(0, Z)

def tanh(Z): 
    return np.tanh(Z)

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, Z):
    return dA * (1 - np.power(tanh(Z), 2))

class CustomNeuralNetwork:
    """
    A handcrafted Neural Network using NumPy.
    Implemented from scratch for educational purposes.
    """
    def __init__(self, layer_dims, activation='ReLU', seed=42):
        np.random.seed(seed)
        self.params = {}
        self.L = len(layer_dims) - 1
        self.activation = activation
        self.cache = {}
        
        # Initialize parameters with He initialization
        for l in range(1, self.L + 1):
            self.params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
            self.params['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
    def compute_loss(self, A_out, Y, loss_fn='Cross-Entropy'):
        m = Y.shape[1]
        A_out = np.clip(A_out, 1e-15, 1 - 1e-15)
        
        if loss_fn == 'Cross-Entropy':
            loss = -1/m * np.sum(Y * np.log(A_out) + (1-Y) * np.log(1-A_out))
        else: # Mean Squared Error
            loss = 1/m * np.sum((A_out - Y)**2)
        return np.squeeze(loss)
        
    def forward_pass(self, X):
        self.cache['A0'] = X
        A = X
        
        # Hidden layers
        for l in range(1, self.L):
            Z = np.dot(self.params['W' + str(l)], A) + self.params['b' + str(l)]
            if self.activation == 'ReLU': 
                A = relu(Z)
            elif self.activation == 'Sigmoid': 
                A = sigmoid(Z)
            elif self.activation == 'Tanh': 
                A = tanh(Z)
            self.cache['Z' + str(l)] = Z
            self.cache['A' + str(l)] = A
            
        # Output layer (Binary classification => Sigmoid)
        Z_out = np.dot(self.params['W' + str(self.L)], A) + self.params['b' + str(self.L)]
        A_out = sigmoid(Z_out)
        self.cache['Z' + str(self.L)] = Z_out
        self.cache['A' + str(self.L)] = A_out
        
        return A_out
        
    def backward_pass(self, Y, A_out, loss_fn='Cross-Entropy'):
        m = Y.shape[1]
        grads = {}
        
        # Derivative of Loss w.r.t final output
        if loss_fn == 'Cross-Entropy':
            dZ_out = A_out - Y
        else: # MSE
            dA_out = 2 * (A_out - Y) / m
            dZ_out = dA_out * sigmoid_backward(np.ones_like(dA_out), self.cache['Z' + str(self.L)])
            
        grads['dW' + str(self.L)] = 1./m * np.dot(dZ_out, self.cache['A' + str(self.L-1)].T)
        grads['db' + str(self.L)] = 1./m * np.sum(dZ_out, axis=1, keepdims=True)
        
        dA = np.dot(self.params['W' + str(self.L)].T, dZ_out)
        
        # Backprop through hidden layers
        for l in reversed(range(1, self.L)):
            if self.activation == 'ReLU': 
                dZ = relu_backward(dA, self.cache['Z' + str(l)])
            elif self.activation == 'Sigmoid': 
                dZ = sigmoid_backward(dA, self.cache['Z' + str(l)])
            elif self.activation == 'Tanh': 
                dZ = tanh_backward(dA, self.cache['Z' + str(l)])
            
            grads['dW' + str(l)] = 1./m * np.dot(dZ, self.cache['A' + str(l-1)].T)
            grads['db' + str(l)] = 1./m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.params['W' + str(l)].T, dZ)
            
        return grads
        
    def update_parameters(self, grads, lr):
        for l in range(1, self.L + 1):
            self.params['W' + str(l)] -= lr * grads['dW' + str(l)]
            self.params['b' + str(l)] -= lr * grads['db' + str(l)]
            
    def predict(self, X):
        A_out = self.forward_pass(X)
        return (A_out > 0.5).astype(int)
