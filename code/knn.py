import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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

class KNN:
	def __init__(self, X, Y, lable):
		self.X = X
		self.Y = Y
		self.lable = lable

	def plot(self, x_axis, y_axis):
		plt.plot(x_axis, y_axis)
		plt.xlabel('K')
		plt.ylabel('Error')
		plt.title(self.lable)
		plt.savefig(self.lable, dpi=600)
		plt.close()

	def evaluateBestK(self):
		plotXAxis, plotYAxis = [], []

		kf = KFold(n_splits=10)

		for k in range(1, 60, 2):
			plotXAxis.append(k)
			plotYAxis.append(0)
			for trainIndex, testIndex in kf.split(self.X):
				X_train, X_test = self.X.loc[trainIndex,], self.X.loc[testIndex,]
				Y_train, Y_test = self.Y.loc[trainIndex,], self.Y.loc[testIndex,]
				classifier = KNeighborsClassifier(n_neighbors=k)
				classifier.fit(X_train, Y_train.values.ravel())
				Y_perdict = classifier.predict(X_test)
				accuracyScore = accuracy_score(Y_perdict, Y_test)
				plotYAxis[-1] += len(Y_test) * (1 - accuracyScore)

			plotYAxis[-1] /= len(self.X)

		self.plot(plotXAxis, plotYAxis)

		minErrorAt = plotYAxis.index(min(plotYAxis))
		minErrorAtK = plotXAxis[minErrorAt]

		return minErrorAtK

if __name__ == '__main__':
	dataSet = DataExtractor('mdata_primary.csv')

	X, Y = dataSet.getPrimaryReclass()
	knnObject = KNN(X, Y, 'PrimaryReclass')
	print('Primary reclass\' best K: {}', knnObject.evaluateBestK())

	X, Y = dataSet.getReclass()
	knnObject = KNN(X, Y, 'Reclass')
	print('Reclass\' best K: {}', knnObject.evaluateBestK())

	X, Y = dataSet.getReclassFirst()
	knnObject = KNN(X, Y, 'ReclassHL')
	print('Reclass only H or L best K: {}', knnObject.evaluateBestK())