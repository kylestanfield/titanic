"""Preprocess the training data for the Kaggle Titanic competition.

Train a tree model on the training data, and then make predictions
for the testing data, and write those predictions to a file.

"""

from sklearn import tree

DATA = open('train.csv', 'r')
X = []
Y = []

def cabin_function(cabin):
    """Take a cabin string, e.g. "B134" and splits it into an array, ['B', '134']"""
    result = []
    if not cabin:
        return [-1, -1]
    if len(cabin) == 1:
        return [ord(cabin[0]), -1]
    if len(cabin) > 4: #If len > 4, there are multiple cabins
        cabin = cabin[0:4]
    result.append(ord(cabin[0])) #Cabin letter
    cabin = cabin[1:]
    if cabin and cabin[0] == ' ':
        #For case when the cabin string is just a letter w/o a number e.g. "F C123"
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
for line in DATA:
    temp = []
    line = line.split(',')
    if line[0].isdigit():
        line[-1] = line[-1][0]
        temp.append(int(line[0])) #Passenger ID
        temp.append(int(line[2])) #Pclass
        if line[5] == 'male': #Gender
            temp.append(0)
        else:
            temp.append(1)
        val = line[6]
        if val:
            temp.append(float(line[6]))
        else:
            temp.append(-1.0)
        temp.append(int(line[7]))
        temp.append(int(line[8]))
        fare = line[10]
        if fare != '':
            temp.append(float(line[10])) #fare
        arr = cabin_function(line[11])
        for item in arr:
            temp.append(item)
        temp.append(ord(line[12]))
        for item in temp:
            if not item:
                item = -1
        X.append(temp)
        Y.append(int(line[1]))

CLASSIFIER = tree.DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5)
CLASSIFIER.fit(X, Y)

DATA.close()

#TEST DATA
DATA = open('test.csv', 'r')
RESULTS_FILE = open('r.txt', 'w')

TEST_DATA = []

for line in DATA:
    temp = []
    line = line.split(',')
    if line[0].isdigit():
        #line[-1] = line[-1][0]
        temp.append(line[0]) #Passenger ID
        temp.append(line[1]) #Pclass
        if line[4] == 'male':
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

        arr = cabin_function(line[10])
        for item in arr:
            temp.append(item)
        temp.append(ord(line[11][0]))
        for item in temp:
            if not item:
                item = -1
        TEST_DATA.append(temp)

PREDICTIONS = CLASSIFIER.predict(TEST_DATA)
RESULTS_FILE.write('PassengerId,Survived\n')
RESULT = ''
for i, item in enumerate(TEST_DATA):
    RESULT = '' + str(item[0]) + ',' + str(PREDICTIONS[i]) +'\n'
    RESULTS_FILE.write(RESULT)
