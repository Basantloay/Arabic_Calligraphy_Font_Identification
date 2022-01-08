# Imports
import numpy as np
from Preproccessing import preproccessing
from FeatureExtraction import extract_features
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# Training Phase
def Phase1(x_train,y_train):
    '''
    Apply Preproccesing and Feature Extraction on  DataSet 
    
    '''
    training_features = np.zeros((len(x_train),255))
    for i in range(0,len(x_train)):
        # Apply Preproccessing
        x_train[i] = preproccessing(x_train[i])
        # Feature Extraction
        training_features[i] = extract_features(x_train[i])
    
    # model fitting
    clf1 = svm.SVC(probability=True, kernel = 'rbf' , gamma=5, C=1000)
    clf2 = MLPClassifier(random_state=0, max_iter=3000, hidden_layer_sizes=[50])
    clf = VotingClassifier(estimators=[('SVM1',clf1),('MLP',clf2)], voting='hard' , weights=[2,1])
    clf.fit(training_features,y_train)

    # return model

    return clf

