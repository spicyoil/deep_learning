
import numpy as np 
import matplotlib.pyplot as plt

def softmax(x):
    maxnum= np.max(x)
    a = np.exp(x-maxnum) #aviod overflow
    b = np.sum(a)
    return a/b
if __name__ == "__main__":
    a = np.array([0.3,0.4,4])
    print(softmax(a))
    print(np.sum(softmax(a)))

    