#HW01.py
#Author: Luke Runyan
#Class: CS7641
#Date: 20220212

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#oneHot Encode an attribute
def OHE_Helper(df, target):
	# Get one hot encoding of atr
	one_hot = pd.get_dummies(df[target])
	# Drop column B as it is now encoded
	df = df.drop(target,axis = 1)
	# Join the encoded df
	df = df.join(one_hot)
	#print(df.head())
	return df  

def encode_Helper(df, target):
	le = LabelEncoder()
	labels = le.fit_transform(df[target])
	df = df.drop(target, axis=1)
	df[target] = labels
	return df

#
#==========================================================================
#
################################### MAIN ###################################
#
#==========================================================================

##########========== PRE-PROCESSING DS1==========##########
#CSV NAME
csv = 'aw_fb_data.csv'
target = ["Running 3 METs",  "Running 5 METs",  "Running 7 METs",  "Self Pace walk",  "Sitting", "Lying"]
x_drop = [target]
df = pd.read_csv(csv)
#pre processing
df = OHE_Helper(df, 'device')
df = OHE_Helper(df,'activity')

# #parse and split
Y = df[target]
X = df.drop(target, axis=1)

# #Normalize Input Data
min_max = MinMaxScaler()
X_Norm = min_max.fit_transform(X)
X = pd.DataFrame(X_Norm)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1, random_state = 42)


'''
classNeuralNetwork(hidden_nodes=None, activation='relu', algorithm='random_hill_climb', 
					max_iters=100, bias=True, is_classifier=True, learning_rate=0.1, 
					early_stopping=False, clip_max=10000000000.0, restarts=0, 
					schedule=<mlrose.decay.GeomDecay object>, pop_size=200, mutation_prob=0.1, 
					max_attempts=10, random_state=None, curve=False)

'''
print()
print("RANDOM OPTIMIZATION FOR NEURAL NETWORKS")
print()
print("__________ GRADIENT DESCENT __________")
# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [18,36], activation = 'relu', \
                                 algorithm = 'gradient_descent', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 10, restarts = 0, max_attempts = 100, \
                                 random_state = 3, curve=False)
start_time = time.time()
nn_model1.fit(x_train, y_train)
training_time = time.time()-start_time
print("Training Time", training_time)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(x_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(x_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy: ", y_test_accuracy)
print()
print()



print("__________ RANDOM HILL CLIMB __________")
# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [18,36], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 500, \
                                 bias = True, is_classifier = True, learning_rate = 0.01, \
                                 early_stopping = True, clip_max = 1.0, restarts = 0, max_attempts = 100, \
                                 random_state = 3, curve=False)
start_time = time.time()
nn_model1.fit(x_train, y_train)
training_time = time.time()-start_time
print("Training Time", training_time)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(x_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(x_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy: ", y_test_accuracy)
print()
print()

print("__________ SIMULATED ANNEALING __________")
# Initialize neural network object and fit object
schedule = mlrose.ExpDecay(init_temp=100, exp_const=0.001, min_temp=0.001)
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [18,36], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 500, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 10, schedule = schedule,
                                  max_attempts = 100, random_state = 3, curve=False)
start_time = time.time()
nn_model1.fit(x_train, y_train)
training_time = time.time()-start_time
print("Training Time", training_time)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(x_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(x_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy: ", y_test_accuracy)
print()
print()



print("__________ GENETIC ALGORITHM __________")
# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [18,36], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 500, \
                                 bias = True, is_classifier = True, learning_rate = 0.0001, \
                                 early_stopping = True, clip_max = 10, pop_size = 200, mutation_prob=0.3,
                                  max_attempts = 100, random_state = 3, curve=False)
start_time = time.time()
nn_model1.fit(x_train, y_train)
training_time = time.time()-start_time
print("Training Time", training_time)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(x_train)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(x_test)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print("Test Accuracy: ", y_test_accuracy)
print()
print()
