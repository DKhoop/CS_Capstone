import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, metrics, linear_model, svm, tree
import os

data = '/Users/diddy/Documents/WGU/Capstone/Project/Ticket_details_modified.csv'
df = pd.read_csv(data)
df['State'] = pd.factorize(df['State'], sort=True)[0] + 1
df['Priority'] = pd.factorize(df['Priority'], sort=True)[0] + 1
df['Category'] = pd.factorize(df['Category'], sort=True)[0] + 1
df['Sub_Category'] = pd.factorize(df['Sub_Category'], sort=True)[0] + 1
df['Skill'] = pd.factorize(df['Skill'], sort=True)[0] + 1
df['Hosting'] = pd.factorize(df['Hosting'], sort=True)[0] + 1
df['Assignee'] = pd.factorize(df['Assignee'], sort=True)[0] + 1

y = df["Days_to_complete"]
X = df.drop(["Days_to_complete"], axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3, random_state=100)

rf_Model = RandomForestClassifier(oob_score=True)
mylog_model = linear_model.LogisticRegression(max_iter=1000)
mysvm_model = svm.SVC(max_iter=1000)
mytree_model = tree.DecisionTreeClassifier()

rf_Model.fit(X_train,y_train)
mylog_model.fit(X_train,y_train)
mysvm_model.fit(X_train,y_train)
mytree_model.fit(X_train,y_train)

y_pred_rf = rf_Model.predict(X_test)
y_pred_log = mylog_model.predict(X_test)
y_pred_svm = mysvm_model.predict(X_test)
y_pred_tree = mytree_model.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred_rf))
print(metrics.accuracy_score(y_test,y_pred_log))
print(metrics.accuracy_score(y_test,y_pred_svm))
print(metrics.accuracy_score(y_test,y_pred_tree))








