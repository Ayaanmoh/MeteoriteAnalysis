import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class DataExtractor:
	def __init__(self, fileName):
		self.dataFrame = pd.read_csv(fileName)

	def getPrimaryReclass(self):
		X = pd.DataFrame()

		dataFrameFeatures = self.dataFrame[['mass', 'year', 'reclat','reclong']]
		for column in dataFrameFeatures.columns:
			maxValue = dataFrameFeatures[column].max()
			minValue = dataFrameFeatures[column].min()
			X[column] = (dataFrameFeatures[column] - minValue) / (maxValue - minValue)

		Y = self.dataFrame['primary_recclass']

		return X, Y

	def getReclass(self):
		X = pd.DataFrame()

		dataFrameFeatures = self.dataFrame[['mass', 'year', 'reclat','reclong']]
		for column in dataFrameFeatures.columns:
			maxValue = dataFrameFeatures[column].max()
			minValue = dataFrameFeatures[column].min()
			X[column] = (dataFrameFeatures[column] - minValue) / (maxValue - minValue)

		Y = self.dataFrame['recclass']

		return X, Y

	def getReclassFirst(self):
		X = pd.DataFrame()

		dataFrameFeatures = self.dataFrame[['mass', 'year', 'reclat','reclong']]
		for column in dataFrameFeatures.columns:
			maxValue = dataFrameFeatures[column].max()
			minValue = dataFrameFeatures[column].min()
			X[column] = (dataFrameFeatures[column] - minValue) / (maxValue - minValue)

		Y = self.dataFrame['recclass'].str[0]

		return X, Y

class DT:
	def __init__(self, X, Y, lable):
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
		self.lable = lable

	def GiniModel(self): 
		errorDataFrame = pd.DataFrame()

		for max_depth in range(3, 100):
			for min_samples_leaf in range(3, 100):
				classifier = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
				classifier.fit(self.X_train, self.Y_train)
				Y_perdict = classifier.predict(self.X_test)
				accuracyScore = accuracy_score(Y_perdict, self.Y_test)
				errorDataFrame.loc[max_depth, min_samples_leaf] = 1 - accuracyScore

		errorDataFrame.to_csv('Gini_' + self.lable + '.csv')

	def EntropyModel(self):
		errorDataFrame = pd.DataFrame()

		for max_depth in range(3, 100):
			for min_samples_leaf in range(3, 100):
				classifier = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
				classifier.fit(self.X_train, self.Y_train)
				Y_perdict = classifier.predict(self.X_test)
				accuracyScore = accuracy_score(Y_perdict, self.Y_test)
				errorDataFrame.loc[max_depth, min_samples_leaf] = 1 - accuracyScore

		errorDataFrame.to_csv('Entropy_' + self.lable + '.csv')

if __name__ == '__main__':
	dataSet = DataExtractor('mdata_primary.csv')

	X, Y = dataSet.getPrimaryReclass()
	dtObject = DT(X, Y, 'PR')
	print('Primary reclass')
	dtObject.GiniModel()
	dtObject.EntropyModel()

	X, Y = dataSet.getReclass()
	dtObject = DT(X, Y, 'R')
	print('\nReclass')
	dtObject.GiniModel()
	dtObject.EntropyModel()

	X, Y = dataSet.getReclassFirst()
	dtObject = DT(X, Y, 'RF')
	print('\nReclass F')
	dtObject.GiniModel()
	dtObject.EntropyModel()