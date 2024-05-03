import sklearn.model_selection as sklMS
import sklearn.linear_model as sklLM
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix

#Question1
file = open('RiceData.data', 'r')
lines = file.readlines()
newLines = []

for i, line in enumerate(lines):
    currLine = line.strip().split(',')
    species = str(currLine[7])
    currLine.pop(7)
    
    #(PART a) Change the target variable to a number.
    if species == 'Cammeo':
        currLine.insert(6,0)
    if species == 'Osmancik':
        currLine.insert(6,1)
    
    s = [float(item) for item in currLine[:-1]]
    s.append(int(currLine[6]))
    newLines.append(s)

#(PART b & c)
#Remove 20% of the examples, selected at random, and keep them for testing
test_data, rest = sklMS.train_test_split(newLines, test_size=0.2, train_size=0.8, random_state=10)

#Split the remaining examples into training (75%) and validation (25%)
validate_data, train_data = sklMS.train_test_split(rest, test_size=0.25, train_size=0.75, random_state=10)

def printConfusionMatrix(data):
    print("     N          |          P")
    print("--------------------------------")
    print("True:   {}      |     {}".format(data[0][0], data[1][1]))
    print("False:  {}      |     {}".format(data[1][0], data[0][1]))

def zeroOneLoss(model, dataset):
    y_pred = []
    y_true = []
    for item in dataset:
        y_pred.append(model.predict([item[:-1]])[0])
        y_true.append(item[-1])
    return zero_one_loss(y_true=y_true, y_pred=y_pred), confusion_matrix(y_true=y_true, y_pred=y_pred)

#Question 4
#logistic regression with regularization
lrr = sklLM.LogisticRegression(penalty='l2')
lrr.fit([item[:-1] for item in train_data], [item[-1]for item in train_data])

#logistic regression with no regularization
lrnr = sklLM.LogisticRegression(penalty=None)
lrnr.fit([item[:-1] for item in train_data], [item[-1]for item in train_data])

lrrZeroOne, lrrConfusion = zeroOneLoss(lrr, validate_data)
lrnrZeroOne, lrnrConfusion = zeroOneLoss(lrnr, validate_data)

print("Logistic regression with regularization Empirical Risk: {}".format(lrrZeroOne))
printConfusionMatrix(lrrConfusion)
print("No Regularization Empirical Risk: {}".format(lrnrZeroOne))
printConfusionMatrix(lrnrConfusion)

