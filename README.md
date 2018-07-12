# ParallelDotsChallenge TV Commercial Detection Dataset made available here for commercial/non-commercial classification problem: https://archive.ics.uci.edu/ml/datasets/TV+News+Channel+Commercial+Detection+Dataset 


This project contains a dataset of 1,29,685 training samples and 4125 features. It is a classification problem to classify whether a sample is a commercial or a non-commercial. The dataset is split into train and test with 97263 and 32422 examples respectively. Further 75000 examples out of training set has been used for training. Following models have been trained and tested :
	1) Support Vector Machine with PCA,LDA
	2) Random Forest Classifier with LDA
	3) andom Forest Classifier

Due to large number of features dimensionality reduction PCA and LDA have been used. Among these models Random Forest Classifier gives highes accuracy of 94.44%. Tuned parameters have been explicitly stated in the test_file.py

The project contains test_file.py which has code to test all the above models and separate files for individual models.
