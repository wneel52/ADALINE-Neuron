# ADALINE style Neuron -> inspired by Widrow-Hoff LMS algorithm and Dr Widrow's youtube lectures
import numpy as np

class adaline_neuron:
    """
    parameters we need are input vector x, weights w, bias b, gradient descent (learning rate) lr
    expected output d, output y and error (d - y)

    x1 -> mul by w1 -> sum
    x2 -> mul by w2 -> sum
    ...                             sum + bias -> output y -> error = d - y -> feedback to weights
    xn -> mul by wn -> sum


    Mathematical representations:
    y = PI w_i*x_i + b
    epsilon = error = d - y
    epsilon^2 = d^2 - 2 x^T w + w^T x x^T w
    MSE = E[epsilon^2] = E[d^2] - 2 w^T E[x d] + w^T E[x x^T] w
    MSE = E[d^2] - 2 w^T p + w^T R w
    where p = E[x d] is the cross-correlation vector and R = E[x x^T] is the autocorrelation matrix

    wj+1 = wj + lr * ( - d(MSE)/d(w) )


    all weights must be updated simultaneously
    """
    def __init__(self):
        self.w = None
        self.b = 0
        self.lr = 0.1
        self.y = 0
        self.d = 0
        self.x = None
        self.error = 0
        self.initialized = False

    def initialize(self, input_size):
        # initialize weights and bias
        self.w = np.random.uniform(-1, 1, input_size)
        self.b = 0.0
        self.initialized = True

    def compute_y(self):
        # compute output y
        #linear_output = np.dot(self.w, self.x) + self.b # compute dot product of weights and inputs plus bias
        #self.y = self.sigmoid(linear_output)  # apply sigmoid function
        self.y = np.dot(self.w, self.x) + self.b # linear activation
        return self.y

    def compute_error(self):
        # compute error
        self.error = self.d - self.y
        return self.error

    def update_weights(self):
        # "take the steepest descent step"
        n = len(self.w)
        for j in range(n):
            self.w[j] = self.w[j] + self.lr * self.error * self.x[j]
        self.b = self.b + self.lr * self.error
        return self.w, self.b

    def training_step(self):
        if not self.initialized:
            raise Exception("Neuron not initialized. Call initialize(input_size) first.")
        self.compute_y()
        self.compute_error()
        self.update_weights()

    def train(self, input_vector, desired_output, epochs=1):
        self.x = input_vector
        self.d = desired_output
        for i in range(epochs):
            self.training_step()
        return self.w, self.b
    
    """ Sigmoid activation function 
        Recommended by Widrow for a better convergence in ADALINE than linear activation
    """
    def sigmoid(self, x):
        return 2 / (1 + np.exp(-x))
