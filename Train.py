from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
import pandas as pd
import joblib

data=pd.read_csv("./diabetes.csv")
print(data.columns)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

trainx, testx, trainy, testy =train_test_split(X, Y, random_state=101, test_size=0.2)

lg=RandomForestClassifier()
lg.fit(trainx, trainy)
y_pred =lg.predict(testx)

classification_report(testy, y_pred)

confusion_matrix(testy, y_pred)
acc=accuracy_score(testy, y_pred)

print(accuracy_score(testy, y_pred))
joblib.dump(lg, "diabeticmodel.pkl")