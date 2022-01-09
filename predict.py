############################ IMPORTS ########################

#Always make all imports in the first cell of the notebook, run them all once.
import numpy as np
from Preproccessing import preproccessing
from FeatureExtraction import extract_features
from IO import *
from Phase1 import *
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score, accuracy_score
import sys
import pickle


# ############################ Read Dataset ########################
'''
1.a) READ THE DATASET
'''
x,y = ReadDataSet()

# # ############################ Phase-1 ########################

model = Phase1(x,y)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# ############################ Phase-2 ########################
# '''

# load the model from disk
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

# 1) Inputs : Full Path to the test-set directory , Full Path to the output directory

inputPath = str(sys.argv[1])
outputPath = str(sys.argv[2])

# 2) Read Test set
# '''
x_test= ReadTestSet(inputPath)

# '''
# 3) Loop on Test set
# '''
y_pred = np.zeros(len(x_test)) # Predicted class for each test image
runningTime = np.zeros(len(x_test)) # Runtime of each test image
# total_start = time.time()
for i in range(0,len(x_test)):
    try:
        #3.a) start timer
        start = time.time()
        #3.b) Preprocess the test image
        x_test[i] = preproccessing(x_test[i].astype('uint8'))
        # 3.c) Extract features from the test image
        x_test[i] = extract_features(x_test[i])
        # 3.d) Classify the test image
        y_pred[i] = model.predict([x_test[i]])

        # 3.e) Stop timer
        runningTime[i] = round(time.time()-start,2)
    except:
        runningTime[i] = round(time.time()-start,2)
        y_pred[i] = -1
        continue

print(len(y_pred))
runningTime[runningTime == 0] = 0.001
WriteOutputFiles(outputPath,y_pred,runningTime)
