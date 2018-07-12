import datetime
import pickle
from  sklearn.datasets import load_svmlight_file as a
from sklearn.externals import joblib
from scipy.sparse import vstack
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier

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


X_train_1=X_train[0:75000]
X_test_1=X_test

print X_train_1.shape

#UNCOMMENT THIS SECTION WHILE TRAINING 
# t_init=datetime.datetime.now()
# print datetime.datetime.now(),"\n"

# rfModel=RandomForestClassifier(max_depth=100,n_jobs=-1,n_estimators=130,verbose=1,max_features=None,oob_score=True).fit(X_train_1,y_train[0:75000])

# t_end=datetime.datetime.now()
# print datetime.datetime.now(),"\n"
# print "Time taken to train ",t_end-t_init
# #print (rfModel.score(X_train_1,y_train[0:75000]))
# joblib.dump(rfModel,'RFmodelFinal.pkl')

# # #############################################

#UNCOMMENT THIS SECTION WHILE TESTING
rfModel=joblib.load('Models/RFmodelFinal.pkl')
t_init=datetime.datetime.now()
print datetime.datetime.now(),"\n"

print (rfModel.score(X_test_1,y_test))
print "Model parameters ",rfModel.get_params(deep=False)

t_end=datetime.datetime.now()
print datetime.datetime.now(),"\n"
print "Time taken to test ",t_end-t_init



