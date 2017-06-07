import inline as inline
import matplotlib
from sklearn.datasets import load_iris
import numpy as np
from scipy import sparse
import pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier

def plot(X_train, y_train):
    fig, ax = plt.subplots(3,3, figsize=(15, 15))
    plt.suptitle("iris_pairplot")
    for i in range(3):
        for j in range(3):
            ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
            ax[i, j].set_xticks(())
            ax[i, j].set_yticks(())
            if i == 2:
                ax[i, j].set_xlabel(iris['feature_names'][j])
            if j == 0:
                ax[i, j].set_ylabel(iris['feature_names'][i + 1])
            if j > i:
                ax[i, j].set_visible(False)


    a = input()

iris = load_iris()
iris.keys()
print(iris['DESCR'][:193] + "\n...")

X_train, X_test, y_treain, y_test = train_test_split(iris['data'],iris['target'],random_state=0)
plot(X_train, y_treain)
pylab.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_treain)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params= None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
y_pred = knn.predict(X_test)
print(np.mean(y_pred == y_test))
print(knn.score(X_test, y_test))