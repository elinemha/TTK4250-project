import numpy as np


ISize = 2
I = np.hstack((np.eye(ISize), np.eye(ISize), np.eye(ISize), np.eye(ISize), np.eye(ISize)))
A = np.eye(2, 10)

B = np.ones((2, 2))*5
C = np.ones((2, 2))*7
D = np.vstack((B, C))
print(D@A)
print(D@I)


print(np.diag(D))