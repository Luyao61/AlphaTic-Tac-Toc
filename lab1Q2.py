import numpy as np
from sklearn import datasets, svm, neighbors

"""
load file
"""
A = np.loadtxt('tictac_multi.txt')
X = A[:,:9]           # Input features
y = A[:,9:]           # Output labels


"""
Linear Regression
"""
def standRegres(x,y):
    #theta = (x.transpose() * x).inverse() * x.transpose() * y
    xt = x.transpose()
    theta = np.linalg.inv(xt.dot(x)).dot(xt).dot(y)
    return theta
theta = standRegres(X,y)
predicted_y = np.zeros(shape = (len(y),9))
for i in range(len(predicted_y)):
    predicted_y[i] = theta.dot(X[i])

#set round up threshold = 0.3
threshold = 0.3
correct = 0.0
for i in range(len(y)):
    for j in range(9):
        if (predicted_y[i,j] >= threshold):
            predicted_y[i,j] = 1
        else:
            predicted_y[i,j] = 0
    if(np.array_equal(predicted_y[i],y[i])):
        correct = correct + 1
accuracy = correct/len(y) * 1.0
print("Linear regression accuracy: %6f"%(accuracy))


"""
K-nearest neighbors
"""
n_neighbors = 5
weights = 'uniform'
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
predicted_y = knn.fit(X, y).predict(X)

#set round up threshold = 0.3
threshold = 0.56
correct = 0.0
for i in range(len(y)):
    for j in range(9):
        if (predicted_y[i,j] >= threshold):
            predicted_y[i,j] = 1
        else:
            predicted_y[i,j] = 0
    if(np.array_equal(predicted_y[i],y[i])):
        correct = correct + 1
accuracy = correct/len(y) * 1.0
print("K-nearest neighbor regressor accuracy: %6f" %(accuracy))


predict = knn.predict(X[1].reshape(1,-1))

