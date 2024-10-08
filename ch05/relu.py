import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
   

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

relu = Relu()

mask = (x <= 0)
print(f"mask check: {mask}")

forward_result = relu.forward(x)
print(f"forward result: {forward_result}")

backward_result = relu.backward(forward_result)
print(f"backward result: {backward_result}")