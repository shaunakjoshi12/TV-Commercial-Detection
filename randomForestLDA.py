import datetime
import pickle
from  sklearn.datasets import load_svmlight_file as a
from sklearn.externals import joblib
from scipy.sparse import vstack
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load preprocessed data
X_train_lda_1=np.loadtxt('Preprocessed_Data/TrainXLDAData.csv',delimiter=',')
X_test_lda_1=np.loadtxt('Preprocessed_Data/TestXLDAData.csv',delimiter=',')

y_train=np.loadtxt('Preprocessed_Data/Preprocessed_Data/TrainYData.csv',delimiter=',')
y_test=np.loadtxt('Preprocessed_Data/TestYData.csv',delimiter=',')

#UNCOMMENT THIS SECTION WHILE TRAINING
# X_train_lda_1=np.expand_dims(X_train_lda_1,axis=1)
# print X_train_lda_1.shape

# t_init=datetime.datetime.now()
# print datetime.datetime.now(),"\n"

# rfModel=RandomForestClassifier(max_depth=500,n_jobs=-1,n_estimators=50,verbose=1,max_features=None,oob_score=True).fit(X_train_lda_1,y_train[0:75000])
# t_end=datetime.datetime.now()
# print datetime.datetime.now(),"\n"
# print "Time taken to train ",t_end-t_init
# #print (rfModel.score(X_train_1,y_train[0:75000]))
# joblib.dump(rfModel,'RFLDAmodelFinal.pkl')

############################################

#UNCOMMENT THIS SECTION WHILE TESTING
lda=joblib.load('Models/RFLDAmodelFinal.pkl')
t_init=datetime.datetime.now()
print datetime.datetime.now(),"\n"

X_test_lda_1=np.expand_dims(X_test_lda_1,axis=1)
print (lda.score(X_test_lda_1,y_test))
print "Model parameters ",lda.get_params(deep=False)

t_end=datetime.datetime.now()
print datetime.datetime.now(),"\n"
print "Time taken to test ",t_end-t_init

