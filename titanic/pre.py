import pandas as pd
from pandas.core.generic import T 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
titanic = pd.read_csv('train.csv')
# titanic.hist(bins=50, figsize=(20, 15))

# corr_matrix = titanic.corr()
# print(corr_matrix)

# attributes = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# scatter_matrix(titanic[attributes], figsize=(12, 8))
# plt.show()

# data cleanning for na value in Age
median = titanic['Age'].median()
med = titanic['Fare'].median()
titanic['Age'].fillna(median, inplace=True)
titanic['Fare'].fillna(med, inplace=True)

# data cleanning for dropping cabin column
titanic = titanic.drop('Cabin', axis=1)
titanic = titanic.drop('Name', axis=1)

titanic = titanic.drop('Ticket', axis=1)
# one-hot encoding
titanic = pd.get_dummies(titanic)

titanic.to_csv('titanic_cld.csv')

test = pd.read_csv('test.csv')
median_test = test['Age'].median()
med_test = test['Fare'].median()
test['Age'].fillna(median_test, inplace=True)
test['Fare'].fillna(med_test, inplace=True)

# data cleanning for dropping cabin column
test = test.drop('Cabin', axis=1)
test = test.drop('Name', axis=1)

test = test.drop('Ticket', axis=1)
# one-hot encoding
test = pd.get_dummies(test)

test.to_csv('test_cld.csv')
