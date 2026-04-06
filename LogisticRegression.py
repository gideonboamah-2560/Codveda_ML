from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.metrics import roc_curve, auc
import numpy as np

dataset = pd.read_csv('c:/Users/LAPTOP/Downloads/churn-bigml-80.csv')


X = dataset.iloc[:,6].values
X = X.reshape(-1, 1)

Y = dataset.iloc[:,19].values

Y = Y.reshape(-1, 1)
bool_array = np.array(Y)
Y_encod = bool_array.astype(int)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encod, test_size= 0.1)

scaler = StandardScaler()
scaler.fit(X_train)

imputer = SimpleImputer(strategy = 'median')

X_train = imputer.fit_transform(X_train)
Y_train = imputer.fit_transform(Y_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


plt.scatter (X, Y, c=Y, cmap ='rainbow')
plt.title ("SCATTER DIAGRAM OF X AND Y")
plt.show()

# Need to optimize the weights for the class_weight***
LR = LogisticRegression(random_state= 40, class_weight= {0: 0.6, 1: 2.4})
LR.fit(X_train, Y_train)
Y_pred = LR.predict(X_test)
Y_pred_prob = LR.predict_proba(X_test)[:,1]

fpr, tpr , thres = roc_curve(Y_test, Y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,)
plt.plot([0, 1], [0, 1])
plt.title(f'ROC Curve (AUC = {roc_auc})')
plt.show()

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)

#f1 = f1_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

LR.intercept_
LR.coef_
manual_log_odds = X_test.dot(LR.coef_.T) + LR.intercept_

print(manual_log_odds.mean())
print(LR.intercept_)
print(LR.coef_)

print(confusion_matrix(Y_test, Y_pred))
print(report)


