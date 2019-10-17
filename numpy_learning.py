import numpy as np 
#basic
x = np.array([1.0,2.0,3.0])
print(x,type(x))
y = np.array([2,4,5])
print(x*y)
print(x*2)

#A= np.array([1,2], [3,4])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A[0])
print(B[0,1])
for x in A:
    for y in x:
        print(y," ")
print(A,A.shape,A.dtype)
print(A+B,"\n",A*B)

#boardcast
A = np.array([[1, 2], [3, 4]])
B = np.array([5, 6])
print (A*B)

#flatten
A = A.flatten()
print(A)

#judgement
print(A>1)
print(A[A>1])