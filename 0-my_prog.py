import numpy as np 
from sklearn import preprocessing, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd 
df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


X_train = np.array(df.drop(['class'],1))
y_train= np.array(df['class'])
X_test = np.array(df2.drop(['class'],1))
y_test= np.array(df2['class'])
#shuffling the data
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#classifier
#clf = svm.SVC()
#clf = neighbors.KNeighborsClassifier()
clf = RandomForestClassifier(n_estimators=100)
#clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print (accuracy)
for line in X_test:

    prediction = clf.predict(np.array(line).reshape(1,-1))
    print (prediction)
