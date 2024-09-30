import numpy as np


def cee(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))
    
# '2'일 확률이 가장 높을 때
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
   
print(cee(np.array(y), np.array(t)))

# '2'일 확률이 가장 높을 때(0번째 값을 가장 크게)
y = [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
   
print(cee(np.array(y), np.array(t)))

# '7'일 확률이 가장 높을 때
y = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.7, 0.0, 0.0]

print(cee(np.array(y), np.array(t)))