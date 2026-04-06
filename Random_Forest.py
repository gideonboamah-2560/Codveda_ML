from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('c:/Users/LAPTOP/Downloads/churn-bigml-80.csv')
data = data.drop(columns =['State', 'International_plan', 'Voice_mail_plan'])
print(data.head())

X = data.iloc[:,0:14]
Y = data.iloc[:,15]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 20)
rf = RandomForestClassifier(
    n_estimators = 1000,
    criterion= 'entropy',
    min_samples_split= 10,
    max_depth= 50,
    max_features= 3

)
rf.fit(X_train, Y_train)
features = pd.DataFrame(rf.feature_importances_, index= X.columns)

Y_pred = rf.predict(X_test)
print(rf.score(X_test, Y_test))
print(classification_report(Y_test, Y_pred))
print(features.head(15))


