import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def perceptron_example():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 0, 1])
    clf = Perceptron()
    clf.fit(X, y)
    predictions = clf.predict(X)
    print(predictions)
    print(clf.predict([[1110,280]]))
perceptron_example()

def accuracy_example():
  scaller = StandardScaler()
  X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
  X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
  X_train_scaled = scaller.fit_transform(X_train)
  X_test_scaled = scaller.transform(X_test)