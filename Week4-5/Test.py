import numpy as np
import pandas as pd
from DecisionTree import *
ds = pd.read_csv('titanic.csv')
cols_to_drop = [
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin',
    'Embarked',
]

df = ds.drop(cols_to_drop, axis=1)
def convert_sex_to_num(s):
    if s=='male':
        return 0
    elif s=='female':
        return 1
    else:
        return s

df.Sex = df.Sex.map(convert_sex_to_num)
data = df.dropna()

input_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
out_cols = ['Survived']

X = data[input_cols]
y = data[out_cols]
#Train:Test:0.8:0.2
slices = int(data.shape[0]*0.8)
training_data = data.head(slices)
Testing_data = data.tail(1-slices)

#Training the data
dt = DecisionTree()
rules = dt.train(training_data,0,{})
#predicting and testing
predicted = []
groundTruth = []
for ix in Testing_data.index:
    predicted.append(dt.predict(rules,Testing_data.loc[ix]))
    
for index, row in Testing_data.iterrows():
    groundTruth.append(int(row['Survived']))

error = 0
for i in range(len(predicted)):
    if(predicted[i] != groundTruth[i]):
        error+=1

print("Accuracy is {0}".format(1 - error/len(predicted)))



