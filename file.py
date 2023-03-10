import pandas as pd
from sklearn.model_selection import train_test_split

Data_set = pd.read_csv("drug_class.csv", delimiter=",")

Features_names= Data_set.columns[0:7]

print(Features_names)

target=Data_set['Drug'].tolist()

target= list(set(target))

print(target)


X = Data_set[['ID', 'Age', 'Sex', 'Blood_Pressure', 'Cholesterol', 'Na_to_K', 'Blood_Oxygen']].values

print(X)

Y = Data_set['Drug']

print(Y)

# Data Preprocessing

from sklearn import preprocessing

import numpy as np

label_gender = preprocessing.LabelEncoder()
label_gender.fit(['Female', 'Male'])
X[:, 2] = label_gender.transform(X[:, 2])

label_BP = preprocessing.LabelEncoder()
label_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 3] = label_BP.transform(X[:, 3])

label_Chol = preprocessing.LabelEncoder()
label_Chol.fit(['NORMAL', 'HIGH'])
X[:, 4] = label_Chol.transform(X[:, 4])

print(X)

print(Data_set.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn import metrics
from sklearn.metrics import confusion_matrix

################KNN###################

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(X_train, y_train)


predicted = neigh.predict(X_test)

print("\nPredicted by KNN",predicted)

results=confusion_matrix(y_test, predicted)

print("\n KNN confusion matrix",results)

print("\nKNN Accuracy: ", metrics.accuracy_score(y_test, predicted))


#################Naive#################

from sklearn.naive_bayes import GaussianNB
#create a GaussianNB Classifier

model=GaussianNB()

#train Model using Training Sets

model.fit(X_train, y_train)


predicted=model.predict(X_test)

print("\nPredicted by Naive",predicted)

results=confusion_matrix(y_test, predicted)
print("\n Naive confusion matrix",results)

print("\nNaive  Accuracy: ", metrics.accuracy_score(y_test, predicted))