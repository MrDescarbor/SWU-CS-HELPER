from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
clf = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2333)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Acc:", clf.score(X_test, y_test))
