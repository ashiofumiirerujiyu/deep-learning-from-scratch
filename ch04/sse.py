import numpy as np


def sse(y, t):
    return 0.5 * np.sum((y-t)**2)
    
# '2'일 확률이 가장 높을 때
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
   
print(sse(np.array(y), np.array(t)))

# '7'일 확률이 가장 높을 때
y = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.7, 0.0, 0.0]

print(sse(np.array(y), np.array(t)))