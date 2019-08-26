import numpy as np
from sklearn import tree

data = open('train.csv', 'r')
X = []
Y = []

def cabinFunction(cabin):
    #Takes in a cabin string, e.g. "B134" and splits it into an array, ['B', '134']
    result = []
    if len(cabin) == 0:
        return [-1, -1]
    if len(cabin) == 1:
        return [ord(cabin[0]), -1]
    if len(cabin) > 4: #If len > 4, there are multiple cabins
        cabin = cabin[0:4]
    result.append(ord(cabin[0])) #Cabin letter
    cabin = cabin[1:]
    
    if cabin != '' and cabin[0] == ' ': #For case when the cabin string is just a letter w/o a number e.g. "F C123"
        result.append(-1)
        return result
        
    index = cabin.find(' ')
    if index >= 0: #If there are multiple cabins, just look at the first one
        num = cabin[:index] #get up to the first space
        result.append(int(num)) #add those digits to the arr
    else: #Otherwise, just add the cabin number
        if cabin != '':
            result.append(int(cabin[:]))
    return result

#TRAINING DATA
for line in data:
    temp = []
    line = line.split(',')
    if(line[0].isdigit()):
        line[-1] = line[-1][0]
        temp.append(int(line[0])) #Passenger ID
        temp.append(int(line[2])) #Pclass
        if(line[5]=='male'): #Gender
            temp.append(0)
        else:
            temp.append(1)
        val = line[6]
        if val != '':
            temp.append(float(line[6]))
        else:
            temp.append(-1.0)
        temp.append(int(line[7]))
        temp.append(int(line[8]))
        fare = line[10]
        if fare != '':
            temp.append(float(line[10])) #fare
        
        arr = cabinFunction(line[11])
        for item in arr:
            temp.append(item)    
     
        temp.append(ord(line[12]))
        for i in range(len(temp)):
            if not (temp[i]):
                temp[i]=-1
        X.append(temp)
        Y.append(int(line[1]))
classifier = tree.DecisionTreeClassifier(random_state=0,max_depth=7)
classifier.fit(X, Y)

data.close()

#TEST DATA
data = open('test.csv', 'r')
resultsfile = open('r.txt', 'w')

testdata = []

for line in data:
    temp = []
    line = line.split(',')
    if(line[0].isdigit()):
        #line[-1] = line[-1][0]
        temp.append(line[0]) #Passenger ID
        temp.append(line[1]) #Pclass
        if(line[4]=='male'):
            temp.append(0)
        else:
            temp.append(1)
        age = line[5]
        if age != '':
            temp.append(float(line[5]))
        else:
            temp.append(-1.0)

        temp.append(int(line[6]))
        temp.append(int(line[7]))
        val = line[9]
        if val != '':
            temp.append(float(line[9]))
        else:
            temp.append(-1.0)

        arr = cabinFunction(line[10])
        for item in arr:
            temp.append(item)
        
        temp.append(ord(line[11][0]))
        for i in range(len(temp)):
            if not (temp[i]):
                temp[i]=-1
        testdata.append(temp)

predictions = classifier.predict(testdata)
resultsfile.write('PassengerId,Survived\n')
for i in range(len(testdata)):
    result = '' + str(testdata[i][0]) + ',' + str(predictions[i]) +'\n'
    resultsfile.write(result) 
