from sklearn.externals import joblib
import datetime
import pickle
import numpy as np
from preprocess import preprocess_RF

print "Loading preprocessed dataset....."
X_train_pca=np.loadtxt('Preprocessed_Data/TrainXPCAData.csv',delimiter=',')
X_test_pca=np.loadtxt('Preprocessed_Data/TestXPCAData.csv',delimiter=',')


X_train_rf,X_test_rf=preprocess_RF()

y_train=np.loadtxt('Preprocessed_Data/TrainYData.csv',delimiter=',')
y_test=np.loadtxt('Preprocessed_Data/TestYData.csv',delimiter=',')

X_train_lda=np.loadtxt('Preprocessed_Data/TrainXLDAData.csv',delimiter=',')
X_test_lda=np.loadtxt('Preprocessed_Data/TestXLDAData.csv',delimiter=',')

X_test_lda=np.expand_dims(X_test_lda,axis=1)

print "Shape of training feature set without dimensionality reduction ",X_train_rf.shape
print "Shape of testing feature set without dimensionality reduction ",X_test_rf.shape

print "Shape of training feature set after PCA ",X_train_pca.shape
print "Shape of testing feature set after PCA ",X_test_pca.shape

print "Shape of training feature set after LDA ",X_train_lda.shape
print "Shape of testing feature set after LDA ",X_test_lda.shape

print "Shape of training labels ",y_train.shape
print "Shape of testing labels ",y_test.shape



while True:
	print 
	num=int(raw_input(" Classification models   : \n 1.SVM applied after Principal Component Analysis (takes 1 min 14 sec to test ) \n 2.Random Forest classifier \n 3.Random Forest applied after Linear Discriminant Analysis \n 4.SVM applied after Linear Discriminant Analysis \n 5.Exit  \n"))

	if num==1: #Accuracy 
		t_init=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		svmModel=joblib.load('Models/SVMmodelPCA_Dim_55_Final.pkl')
		test_score=svmModel.score(X_test_pca,y_test)*100
		t_end=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		print "Time taken to test ",t_end-t_init
		

		print "\n Model parameters ",svmModel.get_params(deep=False)
		print "\n Tuned params: kernel='linear', C=440 "
		print "\n SVM test score :",test_score

	elif num==2:
		t_init=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		rfModel=joblib.load('Models/RFmodelFinal.pkl')

		
		print "\n Model parameters ",rfModel.get_params(deep=False)

		t_end=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		print "Time taken to test ",t_end-t_init

		print "\n Tuned params: n_estimators=130, max_depth=100, max_features=None"
		print "\n Random Forest Test score :",rfModel.score(X_test_rf,y_test)*100

	elif num==3:
		lda=joblib.load('Models/RFLDAmodelFinal.pkl')
		t_init=datetime.datetime.now()
		print datetime.datetime.now(),"\n"

		
		
		print "\nModel parameters ",lda.get_params(deep=False)

		t_end=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		print "Time taken to test ",t_end-t_init
		print "\n Tuned params: n_estimators=50, max_depth=500, max_features=None"
		print "\n Random Forest after LDA Test score: ",lda.score(X_test_lda,y_test)*100

	elif num==4:
		lda=joblib.load('Models/SVMLDAmodelFinal.pkl')
		t_init=datetime.datetime.now()
		print datetime.datetime.now(),"\n"

		

		
		print "Model parameters ",lda.get_params(deep=False)

		t_end=datetime.datetime.now()
		print datetime.datetime.now(),"\n"
		print "Time taken to test ",t_end-t_init
		print "\n Tuned params: kernel='linear'"
		print "\n SVM after LDA Test score : ",lda.score(X_test_lda,y_test)*100

	if num==5:
		exit(0)
