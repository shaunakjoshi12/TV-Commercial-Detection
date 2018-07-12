import numpy as np
from  sklearn.datasets import load_svmlight_file as a
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
import datetime

Dataset1=a("TV_News_Channel_Commercial_Detection_Dataset/BBC.txt",n_features=4125,dtype=np.float64,multilabel=False,zero_based='auto',query_id=False)
Dataset2=a("TV_News_Channel_Commercial_Detection_Dataset/CNN.txt",n_features=4125,dtype=np.float64,multilabel=False,zero_based='auto',query_id=False)
Dataset3=a("TV_News_Channel_Commercial_Detection_Dataset/CNNIBN.txt",n_features=4125,dtype=np.float64,multilabel=False,zero_based='auto',query_id=False)
Dataset4=a("TV_News_Channel_Commercial_Detection_Dataset/NDTV.txt",n_features=4125,dtype=np.float64,multilabel=False,zero_based='auto',query_id=False)
Dataset5=a("TV_News_Channel_Commercial_Detection_Dataset/TIMESNOW.txt",n_features=4125,dtype=np.float64,multilabel=False,zero_based='auto',query_id=False)
# Split each dataset in feature set and labels
bbcx,bbcy=Dataset1
cnnx,cnny=Dataset2
cnnibnx,cnnibny=Dataset3
ndtvx,ndtvy=Dataset4
times_nowx,times_nowy=Dataset5
# Stack features of data entries from all the datasets
data_X=vstack([bbcx,cnnx,cnnibnx,ndtvx,times_nowx]).toarray()
# Stack labels of data entries from all the datasets
Data_Y=np.concatenate((bbcy,cnny,cnnibny,ndtvy,times_nowy), axis=0)
# Split dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(data_X, Data_Y, test_size=0.25,random_state=100)
np.savetxt('TrainYData.csv',y_train,delimiter=',')
np.savetxt('TestYData.csv',y_test,delimiter=',')

def loadandPreprocess():
	# Load all the datasets
	t_init=datetime.datetime.now()
	print datetime.datetime.now(),"\n"

	# Apply l2 normalization to the train and test feature set
	X_train_1=preprocessing.normalize(X_train,norm='l2')
	X_test_1=preprocessing.normalize(X_test,norm='l2')

	# Apply PCA for dimensionality reduction
	pca = RandomizedPCA(n_components=55).fit(X_train_1[0:75000])
	X_train_pca=pca.transform(X_train_1[0:75000])
	print pca.explained_variance_
	print pca.explained_variance_ratio_
	print X_train_pca.shape
	X_test_pca=pca.transform(X_test_1)

	X_train_pca_1=X_train_pca
	X_test_pca_1=X_test_pca


	# Save the preprocessed train and test feature set in csv files
	np.savetxt('TrainXPCAData.csv',X_train_1[0:75000],delimiter=',')
	np.savetxt('TestXPCAData.csv',X_test_1,delimiter=',')

	# Save the labels in csv files
	

	t_end=datetime.datetime.now()	
	print datetime.datetime.now(),"\n"
	print "Time taken to preprocess ",t_end-t_init
	print X_train_1.shape,"\t",X_test_1.shape

def preprocess_RF():
	X_train_1=X_train[0:75000]
	X_test_1=X_test

	# Save the preprocessed train and test feature set in csv files
	# np.savetxt('TrainXRFData.csv',X_train_1,delimiter=',')
	# np.savetxt('TestXRFData.csv',X_test_1,delimiter=',')
	return X_train_1,X_test_1
	

#loadandPreprocess()

preprocess_RF()