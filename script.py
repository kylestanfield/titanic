from __future__ import division
import numpy as np
from sklearn import tree

data = open('train.csv', 'r')
X = []
Y = []

for line in data:
    temp = []
    line = line.split(',')
    if(line[0].isdigit()):
        line[-1] = line[-1][0]
        temp.append(line[0]) #Passenger ID
        temp.append(line[2]) #Pclass
        if(line[5]=='male'):
            temp.append(0)
        else:
            temp.append(1)
        temp.extend(line[6:9])
        temp.append(line[10])
        temp.append(ord(line[12]))
        for i in range(len(temp)):
            if not (temp[i]):
                temp[i]=-1
        X.append(temp)
        Y.append(line[1])
classifier = tree.DecisionTreeClassifier(random_state=0)
classifier.fit(X, Y)

data.close()

data = open('test.csv', 'r')
resultsfile = open('r.txt', 'w')

testdata = []

for line in data:
    temp = []
    line = line.split(',')
    if(line[0].isdigit()):
        line[-1] = line[-1][0]
        temp.append(line[0]) #Passenger ID
        temp.append(line[1]) #Pclass
        if(line[4]=='male'):
            temp.append(0)
        else:
            temp.append(1)
        temp.extend(line[5:8])
        temp.append(line[9])
        temp.append(ord(line[11]))
        for i in range(len(temp)):
            if not (temp[i]):
                temp[i]=-1
        testdata.append(temp)
predictions = classifier.predict(testdata)
print("BEFORE PREDICTIONS\n")
print(predictions)
print("AFTER PREDICTIONS\n")
for i in range(len(testdata)):
    result = '' + testdata[i][0] + ',' + predictions[i] +'\n'
    resultsfile.write(result)
    
