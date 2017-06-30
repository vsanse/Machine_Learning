#machine learning using regression on stock data
import pandas as pd
import quandl as qd
import math, datetime
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')


df=qd.get("WIKI/GOOGL") # getting dataset df=dataframeed
# print df.head()   #prints all columns of datset
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]#features
#creating features

df['HL_PCT']=(df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Close'] * 100.0 #new-close/close
df= df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print df.head()
forcast_col='Adj. Close'

df.fillna(-99999,inplace=True)#to replace Nan values
forcast_out=int(math.ceil(0.01*len(df)))#using data came 10 days ago to predict today
print('days Advance: ',forcast_out)
df['label']=df[forcast_col].shift(-forcast_out)  #label

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression(n_jobs = -1) # for parallel n_jobs= 5,7,8,10..(-1 will run as many job as possible for processor)
#clf = svm.SVR()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
forcast_set = clf.predict(X_lately)
df['Forcast'] = np.nan
print (forcast_set,type(df.iloc[-1].name))
last_date = df.iloc[-1].name
#print (last_date)
#converting to seconds
last_date = last_date.timestamp()
#print (last_date)
one_day = 86400

next_day = last_date + one_day

for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_day)
	next_day += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] 
print (df.tail())
df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()