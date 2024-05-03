import sklearn.model_selection as sklMS
import sklearn.linear_model as sklLM
import numpy as np
from sklearn.metrics import log_loss
import time
import os

def ellipsis():
    #waiting for dramatic effect
    print(".", end='', flush=True)
    time.sleep(1)
    print(".", end='', flush=True)
    time.sleep(1)
    print(".", flush=True)
    time.sleep(1)
    
#Ex1 Data Preprocessing
print("Data Preprocessing, Please Wait", end="", flush=True)
ellipsis()
os.system('cls')
print("Data Loaded!")

file = open('RiceData.data', 'r')
lines = file.readlines()
newLines = []

for i, line in enumerate(lines):
    currLine = line.strip().split(',')
    species = str(currLine[7])
    currLine.pop(7)
    
    if species == 'Cammeo':
        currLine.insert(6,0)
    if species == 'Osmancik':
        currLine.insert(6,1)
    
    s = [float(item) for item in currLine[:-1]]
    s.append(int(currLine[6]))
    newLines.append(s)

#Remove 20% of the examples, selected at random, and keep them for testing
test_data, rest = sklMS.train_test_split(newLines, test_size=0.2, train_size=0.8, random_state=10)
validate_data, train_data = sklMS.train_test_split(rest, test_size=0.25, train_size=0.75, random_state=10)

#convert train_data and validate_data to numpy arrays
train_data = np.array(train_data)
validate_data = np.array(validate_data)


# Excercise 2 Artifical Neural Networks
from sklearn.neural_network import MLPClassifier

print("Artificial Neural Networks:")

# NN with 1 hidden layers of 30 units each
nn_1 = MLPClassifier(hidden_layer_sizes=(30,), random_state=10)
nn_1.fit(train_data[:, :-1], train_data[:, -1])

# NN with 2 hidden layers of 20 units each
nn_2 = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=10)
nn_2.fit(train_data[:, :-1], train_data[:, -1])

# probabilistic predictions & cross-entropy caluclation
nn_1_probs = nn_1.predict_proba(validate_data[:, :-1])
nn_1_cross_entropies = log_loss(validate_data[:, -1], nn_1_probs)

nn_2_probs = nn_2.predict_proba(validate_data[:, :-1])
nn_2_cross_entropies = log_loss(validate_data[:, -1], nn_2_probs)

#Report the cross entropies
print("Cross Entropy for NN with 1 hidden layer:", nn_1_cross_entropies)
print("Cross Entropy for NN with 2 hidden layers:", nn_2_cross_entropies)

#Report better model
if nn_1_cross_entropies < nn_2_cross_entropies:
    print("NN with 1 hidden layer performs better on the validation data.")
else:
    print("NN with 2 hidden layers performs better on the validation data.")

#Excercise 3
#Decision Trees
from sklearn.tree import DecisionTreeClassifier


print('')
print("Decision Trees:")

#parta fit dt
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=10)
dt.fit(train_data[:, :-1], train_data[:, -1])
