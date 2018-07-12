import numpy as np
from  sklearn.datasets import load_svmlight_file as a
from scipy.sparse import vstack
from sklearn import cross_validation 
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing
import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def loadandPreprocess():
	# Load all the datasets
	t_init=datetime.datetime.now()
	print datetime.datetime.now(),"\n"
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
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_X, Data_Y, test_size=0.25,random_state=100)

	# Apply l2 normalization to the train and test feature set
	X_train_1=preprocessing.normalize(X_train,norm='l2')
	X_test_1=preprocessing.normalize(X_test,norm='l2')

	# Apply LDA for dimensionality reduction
	lda = LDA(n_components=3)
	X_train_lda=lda.fit_transform(X_train_1[60000:75000],y_train[60000:75000])
	X_test_lda=lda.transform(X_test_1)
	
	X_train_lda_1=X_train_lda
	X_test_lda_1=X_test_lda

	fhandle=file('TrainXLDAData.csv','a')
	# Save the preprocessed train and test feature set in csv files
	np.savetxt(fhandle,X_train_lda_1,delimiter=',')
	#np.savetxt('TestXLDAData.csv',X_test_lda_1,delimiter=',')

	# Save the labels in csv files
	# np.savetxt('TrainYDataLDA.csv',y_train,delimiter=',')
	# np.savetxt('TestYDataLDA.csv',y_test,delimiter=',')

	t_end=datetime.datetime.now()	
	print datetime.datetime.now(),"\n"
	print "Time taken to preprocess ",t_end-t_init
	print X_train_lda_1.shape,"\t",X_test_lda_1.shape

loadandPreprocess()