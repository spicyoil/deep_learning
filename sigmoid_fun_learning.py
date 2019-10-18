
import numpy as np 
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))

if  __name__ == "__main__":
    
    x = np.array([-1,2.0,4])
    print(sigmoid(x))

    x2= np.arange(-5.0,5.0,0.1)
    y = sigmoid(x2)
    plt.plot(x2,y)
    plt.ylim(-0.1,1.1)
    plt.show()
