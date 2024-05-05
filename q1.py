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
print('')

input_file = "MLHW6\RiceData.data"
mapping = {
    "Cammeo": 0,
    "Osmancik": 1
}

with open(input_file, 'r') as file:
    lines = file.readlines()

rice = []
for _, line in enumerate(lines):
    data = line.strip().split(',')
    typ = data[-1]
    data.pop(-1)
    if typ in mapping:
        data.append(mapping[typ])

    floats = [float(item) for item in data[:-1]]
    floats.append(int(data[-1]))
    rice.append(floats)


#Remove 20% of the examples, selected at random, and keep them for testing
test_data, rest = sklMS.train_test_split(rice, test_size=0.2, train_size=0.8, random_state=10)

train_data, validate_data = sklMS.train_test_split(rest, test_size=0.25, train_size=0.75, random_state=10)

#convert train_data and validate_data to numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)
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
    best_nn = nn_1
    print("NN with 1 hidden layer performs better on the validation data.")
else:
    best_nn = nn_2
    print("NN with 2 hidden layers performs better on the validation data.")

#partc
best_nn.fit(train_data[:, :-1], train_data[:, -1])
test_probs = best_nn.predict_proba(test_data[:, :-1])

#Excercise 3
#Decision Trees
import sklearn.tree as sklTree

print('')
print("Decision Trees:")

#part a fit dt using gini    
dt = sklTree.DecisionTreeClassifier(criterion='gini', max_depth=5)
dt.fit(train_data[:, :-1], train_data[:, -1])

#part b probability predictions
dt_probs = dt.predict_proba(validate_data[:, :-1])
dt_cross_entropies = log_loss(validate_data[:, -1], dt_probs)

#partc fit dt2 using info gain
dt2= sklTree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
dt2.fit(train_data[:, :-1], train_data[:, -1])

#partd probability predictions
dt2_probs = dt2.predict_proba(validate_data[:, :-1])
dt2_cross_entropies = log_loss(validate_data[:, -1], dt2_probs)

#report the cross entropies
print("Cross Entropy for gini dt:", dt_cross_entropies)
print("Cross Entropy for info gain dt:", dt2_cross_entropies)

if dt_cross_entropies < dt2_cross_entropies:
    best_dt = dt
    print("Decision Tree with gini performs better on the validation data.")
else:
    best_dt = dt2
    print("Decision Tree with info gain performs better on the validation data.")

#Excercise 4 Boosting
from sklearn.ensemble import AdaBoostClassifier as ada

print('')
print("Boosting:")

b20 = ada(n_estimators=20, algorithm='SAMME')
b20.fit(train_data[:, :-1], train_data[:, -1])

b40 = ada(n_estimators=40, algorithm='SAMME')
b40.fit(train_data[:, :-1], train_data[:, -1])

b60 = ada(n_estimators=60, algorithm='SAMME')
b60.fit(train_data[:, :-1], train_data[:, -1])

b20_probs = b20.predict_proba(validate_data[:, :-1])
b40_probs = b40.predict_proba(validate_data[:, :-1])
b60_probs = b60.predict_proba(validate_data[:, :-1])

b20_cross_entropies = log_loss(validate_data[:, -1], b20_probs)
b40_cross_entropies = log_loss(validate_data[:, -1], b40_probs)
b60_cross_entropies = log_loss(validate_data[:, -1], b60_probs)

print("Cross Entropy for boosting with 20 estimators:", b20_cross_entropies)
print("Cross Entropy for boosting with 40 estimators:", b40_cross_entropies)
print("Cross Entropy for boosting with 60 estimators:", b60_cross_entropies)

print("Boosting with 20 estimators performs better on the validation data.")