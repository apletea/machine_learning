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


def scaller_example():
  scaller = StandardScaler()
  X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
  X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
  X_train_scaled = scaller.fit_transform(X_train)
  X_test_scaled = scaller.transform(X_test)


def main():
    #loading data
    data_test = pandas.read_csv('perceptron-test.csv', header=None)
    data_train = pandas.read_csv('perceptron-train.csv', header=None)
    #parseing data
    X_train = data_train.loc[:,1:2]
    Y_train = data_train[0]

    X_test  = data_test.loc[:,1:2]
    Y_test = data_test[0]
    #init and train modekl
    clf = Perceptron(random_state=241)
    clf.fit(X_train,Y_train)
    #predict for objects
    y_pred = clf.predict(X_test)
    #count score
    fist_ans = (accuracy_score(Y_test, y_pred))
    #init new perceptron
    clf = Perceptron(random_state=241)
    #init scaller
    scaller = StandardScaler()
    #scale data
    X_train_scalled = scaller.fit_transform(X_train)
    X_test_scalled = scaller.transform(X_test)
    #fit new model
    clf.fit(X_train_scalled,Y_train)
    #predict for new objects
    y_pred = clf.predict(X_test_scalled)
    #find new accuracy
    second_ans = accuracy_score(Y_test,y_pred)
    print(second_ans-fist_ans)


main()