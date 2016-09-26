import numpy
from sklearn.cross_validation import KFold
from sklearn import datasets, svm, neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix



data_set = ['tictac_final.txt','tictac_single.txt']
classifier = ['linearSVM', 'k-nearest neighbors', 'multilayer perceptron']


for ds in data_set:
    if ds == 'tictac_final.txt':
        cmSize = 2
    elif ds == 'tictac_single.txt':
        cmSize = 9
    print('Data Set: ' + ds +'\n')
    A = numpy.loadtxt(ds)

    ##numpy.random.shuffle(A)
    X = A[:,:9]           # Input features
    y = A[:,9:]           # Output labels


    for cf in classifier:
        if cf == 'linearSVM':
            clf = svm.SVC(kernel='linear')
        elif cf == 'k-nearest neighbors':
            clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
        elif cf == 'multilayer perceptron':
            clf = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        print('Using ' + cf+'\n')
        kf = KFold(len(X), n_folds=10,shuffle=True)

        Accuracy = 0
        cm = numpy.zeros((cmSize,cmSize))
        for train_index, test_index in kf:

            train_y = numpy.ravel(y[train_index])
            clf.fit(X[train_index],train_y)
            
            score = clf.score(X[test_index],y[test_index])
            Accuracy += score

            predict_y = clf.predict(X[test_index])
            cm = cm + confusion_matrix(y[test_index], predict_y)
            

        Accuracy = Accuracy/10
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

        print("Accuracy of " + cf + ' on dataset: ' + ds + ' is %.6f' %(Accuracy))
        print("Confusion Matrix of " + cf + ' on dataset: ' + ds + ' is:')
        print(cm_normalized)
        print('\n')

