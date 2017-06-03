import numpy as np
import pandas
import heapq
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold


#parametrs brutforce
def parametrs():
    X={}
    y = {}
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)

def data_extracting():
    newsgroups = datasets.fetch_20newsgroups(subset='all',categories=['alt.atheism','sci.space'])

def diference_parameters(X1, y):
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X1, y[0])
    for a in gs.grid_scores_:
        print(a.mean_validation_score, " - ", a.parameters)


def main():
    data = datasets.fetch_20newsgroups(subset='all',categories=['alt.atheism','sci.space'])

    X = pandas.DataFrame(data.data)
    y = pandas.DataFrame(data.target)
    #transormation tfdif
    vectorize = TfidfVectorizer(min_df=1)
    #matrix where to each row -> word
    X1 = vectorize.fit_transform(X[0])
    #names of fetch-rows = words
    fn = vectorize.get_feature_names()

    #creating classifier
    clf = SVC(random_state=241,kernel='linear',C=1.0)
    #learning classifier
    clf.fit(X1,y[0])

    #classifer weights
    c = clf.coef_.toarray()
    #absolute value of weights
    cabs = [abs(number) for number in c]
    #searching for rop10 words
    top10 = heapq.nlargest(10,enumerate(cabs[0]),key=lambda x: x[1])
    #list of words
    words = list()
    for e in top10:
        words.append(fn[e[0]])
    #sort words by alphabet
    print(fn)
    words.sort()

    result = ','.join(words)
    print(result)

main()