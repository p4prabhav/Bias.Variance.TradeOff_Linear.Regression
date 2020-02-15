import pickle        
import csv
import matplotlib.pyplot as plt
import base64 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.patches as mpatches
plt.style.use('seaborn-whitegrid')
from six.moves import cPickle as pickle 
from sklearn import preprocessing, svm 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

file=open("./Q1_data/data.pkl","rb")
data=pickle.load(file)
file.close()

np.random.shuffle(data)

X = data[:,0]
Y = data[:,1]
seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)

X_train = X_train.T
X_train = X_train.reshape(-1, 1)

X_test = X_test.T
X_test = X_test.reshape(-1, 1)

X_strain = np.split(X_train,10)
Y_strain = np.split(Y_train,10)


bias2 = [0]*9
bias = [0]*9
var = [0]*9
terror = [0]*9
y_p = [[]]*10
x_axis = [0]*9
avg = 0
l1 = []


for j in range(1,10,1):
	pol = PolynomialFeatures(degree=j)
	for i in range(10):
		model = LinearRegression()
		X_train_ft = pol.fit_transform(X_strain[i])
		X_test_ft = pol.fit_transform(X_test)
		model.fit(X_train_ft, Y_strain[i])
		y_p[i] = model.predict(X_test_ft)
		avg += y_p[i]

	avg = avg/10
	x_axis[j-1] = j

	sq = ((abs(avg-Y_test))**2)
	bias2[j-1] = np.mean(sq)
	bias[j-1] = np.sqrt(bias2[j-1])
	var[j-1] = np.mean(np.var(y_p, axis = 0))
	terror[j-1] = var[j-1] + bias[j-1]**2
	l1.append(j)
data = {'Degree':x_axis,
		'Bias' : bias,
		'Bias^2': bias2,
		'Variance':var,
		}
df = pd.DataFrame(data)
print(df)