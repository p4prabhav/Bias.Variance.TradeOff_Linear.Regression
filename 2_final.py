import pickle        
import csv
import base64 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('seaborn-whitegrid')
from six.moves import cPickle as pickle 
from sklearn import preprocessing, svm 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
np.set_printoptions(suppress=True)

X_train = pickle.load(open('./Q2_data/X_train.pkl', 'rb'))
Y_train = pickle.load(open('./Q2_data/Y_train.pkl', 'rb'))
X_test = pickle.load(open('./Q2_data/X_test.pkl', 'rb'))
Y_test = pickle.load(open('./Q2_data/Fx_test.pkl', 'rb'))

X_test = np.expand_dims(X_test, axis = 1)


bias = [0]*9
var = [0]*9
terror = [0]*9
y_p = [[]]*20
x_axis = [0]*9
l1 = []


for j in range(1,10,1):
	sq = 0
	pol = PolynomialFeatures(degree=j)	
	avg = 0
	for i in range(20):
		model = LinearRegression()
		t = X_train[i]
		t = np.expand_dims(t, axis = 1)
		X_fttrain = pol.fit_transform(t)
		X_fttest = pol.fit_transform(X_test)
		model.fit(X_fttrain, Y_train[i])
		y_p[i] = model.predict(X_fttest)
		avg += y_p[i]

	avg = avg/20
	l1.append(j)
	x_axis[j-1] = j

	sq = ((avg-Y_test))**2
	bias[j-1] = np.mean(sq)
	var[j-1] = np.mean(np.var(y_p, axis = 0))
	terror[j-1] = var[j-1] + bias[j-1]**2

plt.plot(l1, var, color='black')
plt.plot(l1, bias, color="red")
plt.xlabel('Degree')
plt.ylabel('Errors')
red_patch = mpatches.Patch(color='red', label='Bias Square')
black_patch = mpatches.Patch(color='black', label='Variance')
plt.legend(handles=[red_patch,black_patch])
plt.show()

data = {'Degree':x_axis,
		'Variance':var,
		'Bias^2':bias,
		}
df = pd.DataFrame(data)
print(df)