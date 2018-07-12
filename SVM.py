import datetime
import pickle
from  sklearn.datasets import load_svmlight_file as a
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np


# Load preprocessed data
X_train_pca_1=np.loadtxt('Preprocessed_Data/TrainXPCAData.csv',delimiter=',')
X_test_pca_1=np.loadtxt('Preprocessed_Data/TestXPCAData.csv',delimiter=',')

y_train=np.loadtxt('Preprocessed_Data/Preprocessed_Data/TrainYData.csv',delimiter=',')
y_test=np.loadtxt('Preprocessed_Data/TestYData.csv',delimiter=',')



# Print shapes of train and test data
print X_train_pca_1.shape
print X_test_pca_1.shape
print y_train.shape
print y_test.shape

#UNCOMMENT THIS SECTION WHILE TRAINING

# Train SVM with LINEAR kernel 
# t_init=datetime.datetime.now()
# print datetime.datetime.now(),"\n"
# svmModel=svm.SVC(C=440,verbose=True,kernel='linear',decision_function_shape='ovr',tol=0.001).fit(X_train_pca_1,y_train[0:75000])
# train_score=svmModel.score(X_train_pca_1,y_train[0:75000])*100


# t_end=datetime.datetime.now()
# print datetime.datetime.now(),"\n"
# print "Time taken to train ",t_end-t_init
# print ("SVM train score ",train_score)

# # # Save model
#  joblib.dump(svmModel,'SVMmodelPCA_Dim_55_Final.pkl')
##########################################


#UNCOMMENT THIS SECTION WHILE TESTING

# Test SVM model
t_init=datetime.datetime.now()
print datetime.datetime.now(),"\n"
svmModel=joblib.load('Models/SVMmodelPCA_Dim_55_Final.pkl')
test_score=svmModel.score(X_test_pca_1,y_test)*100
t_end=datetime.datetime.now()
print datetime.datetime.now(),"\n"
print "Time taken to test ",t_end-t_init
print ("SVM test score ",test_score)








