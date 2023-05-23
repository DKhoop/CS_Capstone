import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection, metrics, linear_model, svm, tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay
import time
import numpy as np
from numpy import mean
from numpy import std
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = '/Users/diddy/Documents/WGU/Capstone/CyberSure/Cybersure_IT_HelpDesk_Tickets.csv'
df = pd.read_csv(data)

y = df.values[:, 7]
X = df.values[:, :6]

#rf_Model = RandomForestClassifier(oob_score=True)
# #mylog_model = linear_model.LogisticRegression(max_iter=5000)
#mysvm_model = svm.SVC(max_iter=5000)
# #mytree_model = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,random_state=42, test_size=0.3)

print(X_train.shape, y_train.shape)
# identify outliers in the training dataset
iso = IsolationForest(contamination=0.1)
yhat = iso.fit_predict(X_train)
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]
print(X_train.shape, y_train.shape)
# fit the model
model = svm.SVC(max_iter=5000)
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)
print(metrics.accuracy_score(y_test,yhat))

#rf_Model.fit(X_train,y_train)
# #mylog_model.fit(X_train,y_train)
#mysvm_model.fit(X_train,y_train)
# #mytree_model.fit(X_train,y_train)
# #
#y_pred_rf = rf_Model.predict(X_test)
# #y_pred_log = mylog_model.predict(X_test)
#y_pred_svm = mysvm_model.predict(X_test)
# #y_pred_tree = mytree_model.predict(X_test)
# #
# print(metrics.accuracy_score(y_test,y_pred_rf))
# #print(metrics.accuracy_score(y_test,y_pred_log))
#print(metrics.accuracy_score(y_test,y_pred_svm))
# #print(metrics.accuracy_score(y_test,y_pred_tree))
# #ConfusionMatrixDisplay.from_predictions(y_test,y_pred_svm)








