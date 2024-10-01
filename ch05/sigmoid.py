import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))    

    def forward(self, x):
        out = self.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
   

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

sigmoid = Sigmoid()

forward_result = sigmoid.forward(x)
print(f"forward result: {forward_result}")

backward_result = sigmoid.backward(forward_result)
print(f"backward result: {backward_result}")