import numpy
A = numpy.loadtxt('tictac_final.txt')

##numpy.random.shuffle(A)
X = A[:,:9]           # Input features
y = A[:,9:]           # Output labels
##X = [0,1,2,3,4,5,6,7,8,9]
##y = [9,8,7,6,5,4,3,2,1,0]


##numFolds = 10
##X_folds = numpy.array_split(X, numFolds)
##y_folds = numpy.array_split(y, numFolds)
"""
for i in range(10):
    print len(X_folds[i])
    print len(y_folds[i])
"""
from sklearn.cross_validation import KFold
from sklearn import datasets, svm

clf = svm.SVC(kernel='linear', C=1)
kf = KFold(len(X), n_folds=10,shuffle=True)

count = 0
for train_index, test_index in kf:
    
    train_y = numpy.ravel(y[train_index])
    clf.fit(X[train_index],train_y)
    ##test_y = numpy.ravel(y[test_index])
    score = clf.score(X[test_index],y[test_index])
    
    ##print X[train_index].shape,y[train_index].shape
    ##print X[test_index].shape, y[1].shape

    print "Validation Accuracy of fold = %d is %.6f" %(count,score)
    count = count + 1


