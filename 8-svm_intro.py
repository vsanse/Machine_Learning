import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd 
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#df = pd.read_csv('test.data')
df.replace('?', -99999, inplace = True)
#removing useless columns like id
df.drop(['id'], 1, inplace = True) 
#print (df)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

#shuffling the data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#classifier
clf = svm.SVC()
#clf = neighbors.KNeighborsClassifier()
#clf = RandomForestClassifier(n_estimators=1)
#clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print (accuracy)

#testing agains unknown/manual data data
example_measures = np.array([[1,2,5,6,1,7,8,2,4], [3,4,1,4,6,2,3,1,4]])
#reshape to bring data in shape as required by classifier
example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)

print (prediction)