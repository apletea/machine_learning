import pandas
import os
from sets import  Set

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
labeled_data = pandas.read_csv('train.csv', index_col='Id')


languages = Set()
languages_a = []
for val in labeled_data['Language']:
    languages.add(val)
    languages_a.append(val)
languages_le = preprocessing.LabelEncoder()
languages_le.fit(list(languages))
languages_a = languages_le.transform(languages_a)

countries = Set()
countries_a = []
for val in labeled_data['Country']:
    countries.add(val)
    countries_a.append(val)
countries_le = preprocessing.LabelEncoder()
countries_le.fit(list(countries))
countries_a = countries_le.transform(countries_a)

ratings = Set()
ratings_a = []
for val in labeled_data['Rating']:
    ratings.add(val)
    ratings_a.append(val)
ratings_le = preprocessing.LabelEncoder()
ratings_le.fit(list(ratings))
ratings_a = ratings_le.transform(ratings_a)

labeled_data['Language'],labeled_data['Country'],labeled_data['Rating'] = languages_a,countries_a,ratings_a

data = labeled_data.iloc[0:3635, 1:26]
#data = preprocessing.scale(data)
print data
labels = labeled_data.iloc[0:3635, :1]
train_data, test_data, train_labeles, test_labeles = train_test_split(data,labels,train_size=0.9,random_state=250)

clf = MLPClassifier(activation='relu',batch_size=200,random_state=247)
#clf = DecisionTreeClassifier(random_state=247)
#clf = svm.SVC(random_state=247)
#clf = QuadraticDiscriminantAnalysis()
#clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
clf.fit(train_data,train_labeles.values.ravel())
print clf.predict_proba(test_data)[:,:1]

print clf.score(test_data,test_labeles)
