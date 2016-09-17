import numpy
A = numpy.loadtxt('tictac_final.txt')

numpy.random.shuffle(A)
X = A[:,:9]           # Input features
y = A[:,9:]           # Output labels

numFolds = 10
X_folds = numpy.array_split(X, numFolds)
y_folds = numpy.array_split(y, numFolds)

for i in range(10):
    print len(X_folds[i])
    print len(y_folds[i])

from sklearn import datasets, svm



