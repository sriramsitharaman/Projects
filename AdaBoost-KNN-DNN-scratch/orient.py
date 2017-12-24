#!/usr/bin/env python
'''
Model file to be used
----------------------
best : nnet_model_file.npy
nnet : nnet_model_file.npy
adaboost : adaboost_model_file.txt
knn : knn_model_file.txt

K Nearest Neighbors Algorithm:
------------------------------
The algorithm for KNN is as follows

Step 1: Train data was read and the feature vectors are created
Step 2: For each data point in the test data 
	a. compute the K closest neighbors to the data point based on a distance metric.
	b. Take the classes of the K data points and find the majority class amaong them and assign it to the test data point.


A series of Experiments were conducted to arrive at the best configuration for the KNN algorithm. 
Three different distance metrics were tried. Euclidean, Manhattan and Mahalanobis diatance. In addition, Euclidean distance weighted prediction 
was also tried. The Euclidean distance with k=47 and the Manhattan distance with k=67 gave a similar accuracy of 71.68% which was also the
best performance achieved by KNN. However Euclidean distance where k=47 was selected based on smaller runtime.


Adaboost Algorithm:
-------------------
The algorithm given in Freund & Schapire was used for implementing Adaboost. 

Step 1: 6 Classifiers were built for identifying:
	Orientation 0 vs Orientation 90 
	Orientation 0 vs Orientation 180 
	Orientation 0 vs Orientation 270 
	Orientation 90 vs Orientation 180 
	Orientation 90 vs Orientation 270 
	Orientation 180 vs Orientation 270 
Step 2: For each classifier, 20 stumps were built. Each stump compares the value of two pixels. 192 x 191 stumps 
were created to compare if pixel_i >= pixel_j
Step 3: To choose the stumps, the weighted error was calculated for each of the stumps and the stump with least 
weighted error was chosen. The weights of the images are initialized based on positive and negative sample distribution
Step 4: After choosing the first stump, prediction is done on the images and correctly classified images are down-weighted. 
The weights are then normalized, and the subsequent stumps are created in a similar manner
Step 5: The pixels compared by the stumps and the alpha value corresponding to the stumps are stored
Step 6: Other 5 classifiers are built in a similar manner, taking the same number of images as input and the same 
number of stumps
Step 7: The data label is predicted by each stump of the classifier. This predicted vector is multiplied by the 
alpha value of the corresponding stump. The vectors produced by the 20 stumps are added to generate a final vector for the classifier
Step 8: Each of the 6 classifiers predictions are swapped to generate predictions for the opposite classes 
Step 9: The final prediction is done by comparing the 12 vectors produced in Step 7. The class corresponding 
to maximum value is chosen as the predicted class


Neural Network Model:
--------------------
Feed forward fully connected neural network with hidden layers varying from 1 to 5 along with backpropagation was implemented . A momemtum update was also performed while updating the 
weights and biases after the gradient calculation. Weights of the network are randomly initialized between -1 to 1. Any other initialization resulted in 
slower convergence. 

Also , the neural network was trained for 25 epochs and in each epoch the training dataset is shuffled before use.

Input data is 192*1 feature array
Output data 4*1 array (activation corresponding to 0,90,180,270)
Best Model:
5 Hidden layer NN with (256,192,128,64,32) neurons in layer 1-5 respectively
Learning rate : 0.1
Momentum rate : 0.05

model output File :It  is a numpy array of weights and biases . It should be stored in ".npy" format
'''


import sys,os
import numpy as np
from random import shuffle
import time 
import json
from collections import Counter
import pandas as pd
from operator import itemgetter


def sigmoid(x):
    return 1/(1+np.exp(-1.0*np.clip(x,-100,100)))
def sigmoidDer(x):
    return sigmoid(x)*(1-sigmoid(x))
def cost(m,n):
    diff=[m[i].reshape(4)-n[i] for i in range(len(m))]
    return sum([0.5*sum(i*i) for i in diff])
def onehotencode(y):
    max_y=3
    one_hot_y=[]
    map_dict={0:0,90:1,180:2,270:3}
    for i in range(len(y)):
        temp=[0]*(max_y+1)
        temp[map_dict[y[i]]]=1
        one_hot_y.append(temp)
    return one_hot_y

def forward_prop(numLayers,activations,before_actvs,nn):
    activation=activations[0]
    for layer in range(1,numLayers+2):
        before_actv=np.dot(nn.weights[layer-1],activation)+nn.bias[layer-1]
        before_actvs.append(before_actv)
        activation=sigmoid(before_actv)
        activations.append(activation)
    return activations,before_actvs

class NNStructure:
    def __init__(self,inputData,numLayers,neuronDict,uniqueClasses):
        self.weights=[]
        self.bias=[]
        for i in range(numLayers+1):
            if i==0:
                np.random.seed(42)
                self.weights.append(2*np.random.random((neuronDict[i+1],len(inputData[0])))-1)
                np.random.seed(42)
                self.bias.append(2*np.random.random((neuronDict[i+1],1))-1)
            elif i==numLayers:
                np.random.seed(42)
                self.weights.append(2*np.random.random((uniqueClasses,neuronDict[i]))-1)
                np.random.seed(42)
                self.bias.append(2*np.random.random((uniqueClasses,1))-1)
            else:
                np.random.seed(42)
                self.weights.append(2*np.random.random((neuronDict[i+1],neuronDict[i]))-1)
                np.random.seed(42)
                self.bias.append(2*np.random.random((neuronDict[i+1],1))-1)

#Will train a neural net model for 5 hidden layers , learning rate 0.1 and momentum rate 0.05 
def trainNN(trainFile):
    X_train=[]
    Y_train=[]
    with open(trainFile,"r") as inf:
        for row in inf:
            cols=row.split(" ")
            X_train.append([1.0*int(i) for i in cols[2:]])
            Y_train.append(int(cols[1]))
    inf.close()
    layerDict={1:256,2:192,3:128,4:64,5:32}
    numLayers=5
    learning_rate=0.1
    mu=0.05
    nn=NNStructure(X_train,numLayers,layerDict,4)
    Y_train_oh=onehotencode(Y_train)
    inputIndices= [i for i in range(len(X_train))]
    cost1=[]
    accuracy=[]
    percent=0
    grad_b=[np.zeros(b.shape) for b in nn.bias]
    grad_w=[np.zeros(w.shape) for w in nn.weights]
    bestTestAccuracy=0
    bestweights=0
    bestBias=0
    perf_dict={}
    for epoch in range(15):
        error=0
        #start_time=time.time()
        #print ("Epoch",epoch)
        predictions=[]
        shuffle(inputIndices)
        numRowsData=len(inputIndices)
        count=0
        for i in inputIndices:
			# Forward propagation
            prev_grad_b=grad_b
            prev_grad_w=grad_w
            x=X_train[i] 
            y=Y_train_oh[i]
            curr_activation=np.array(x).reshape(len(x),1)
            layeractivations=[curr_activation]
            before_actvs=[]
            layeractivations,before_actvs=forward_prop(numLayers,layeractivations,before_actvs,nn)
            predictions.append(layeractivations[-1])
            delta=-1*sigmoidDer(before_actvs[-1])*(layeractivations[-1]-np.array(y).reshape(4,1))
            grad_b[-1]=delta
            grad_w[-1] = np.dot(delta, layeractivations[-2].transpose())
			#backpropagation
            for layer in range(numLayers,0,-1):
                before_actv=before_actvs[layer-1]
                sig_der=sigmoidDer(before_actv)
                delta = np.dot(nn.weights[layer].transpose(), delta)* sig_der
                grad_b[layer-1] = delta
                grad_w[layer-1] = np.dot(delta, layeractivations[layer-1].transpose())
            for layer in range(numLayers):
                nn.weights[layer]+=(grad_w[layer]*learning_rate+mu*prev_grad_w[layer])
                nn.bias[layer]+=(grad_b[layer]*learning_rate+mu*prev_grad_b[layer])
    return nn.weights,nn.bias

def predict_nnet(inputFile,nnet_model):
    X_test=[]
    Y_test=[]
    test_images_id=[]
    with open(os.getcwd()+"\\"+inputFile,"r") as inf:
        for row in inf:
            cols=row.split(" ")
            test_images_id.append(cols[0])
            X_test.append([1.0*int(i) for i in cols[2:]])
            Y_test.append(int(cols[1]))
    inf.close()
    X_test=(X_test-np.mean(X_test,axis=0))/np.std(X_test,axis=0)
    Y_test_oh=onehotencode(Y_test)
    
    pred_weight=nnet_model[0]
    pred_bias=nnet_model[1]
    correct=0
    predictions=[]
    indices_correctness=[]
    check=[0,90,180,270]
    outputFile=open("output.txt","w")
    for i in range(len(X_test)):
        image_id=test_images_id[i]
        x=X_test[i]
        y=Y_test[i]
        current_activation=np.array(x).reshape(len(x),1)
        layeractivations=[current_activation]
        zs=[]
        current_activation=layeractivations[0]
        numLayers=5
        for layer in range(1,numLayers+2):
            z=np.dot(pred_weight[layer-1],current_activation)+pred_bias[layer-1]
            zs.append(z)
            current_activation=sigmoid(z)#if layer==numLayers+1 else np.maximum(0,z)
            layeractivations.append(current_activation)#*drop)
        maxIndex=np.argmax(layeractivations[-1])
        pred_oh=[1 if x==maxIndex else 0 for x in range(4)]
        #predictions.append(check[maxIndex])
        correct+=1 if pred_oh==Y_test_oh[i] else 0
        indices_correctness.append(1 if check[maxIndex]==y else 0)
        outputFile.write(" ".join([image_id,str(check[maxIndex])])+"\n")
    outputFile.close()
    print "Accuracy: ",1.0*sum(indices_correctness)/len(X_test)*100.0
    #return predictions,indices_correctness

	
### KNN Code Starts Here

def trainKNN(TrainfileName,modelFileName):

	# Read the training data line by line
	feature_dict = {}
	index =0
	with open(TrainfileName,'r') as File:
		for line in File:
			data = line.strip().split(" ")
			#stroring the feature set and the correct orientation for every line in the train data.
			feature_dict[index]= (data[2::],data[1]);
			index+=1
	File.close()
	# Write the dictionary of features to the model file
	with open(modelFileName, 'w') as file:
		file.write(json.dumps(feature_dict))

def testKNN(TestfileName,modelFileName):
	d2 = json.load(open(modelFileName))
	feature_dict = eval('d2')
	feature_dict = {int(key): value for key, value in feature_dict.iteritems()}
	feature_array = np.asarray( map( lambda x: map(int,x[0]), feature_dict.values()))
	test_feature_dict = {}
	label = {}
	imageName =[]
	index =0
	# Read the test file
	with open(TestfileName,'r') as File:
		for line in File:
			testData = line.strip().split(" ")
			testFeature = testData[2::]
			test_feature_dict[index] = testFeature
			label[index]=testData[1]
			imageName.append(testData[0])
			index+=1
	# create the feature array for the test file
	test_array = np.asarray(  map( lambda x: map(int,x) , test_feature_dict.values()))
	
	output_file = 'output.txt'
	opfile = open(output_file, 'w')
	# compute the K=10 nearest neighbors:
	k=47
	# Fucntion that evaluates Euclidean distance
	def evaluateDistance(a):
		return np.sqrt(np.sum(np.square(np.subtract(feature_array,  np.asarray(map(int,a)))),axis=1))
		
	# Fuction to Sort the numpy array
	def sortArray(a):
		return np.argsort(a)
	#Fuction to get the max vote of the K nearest neighbors
	def vote(x):
		return int(Counter([feature_dict[ind][1] for ind in x]).most_common(1)[0][0])
	
	# Numpy array that stores the distances of all the 
	distnaceArray = np.apply_along_axis(evaluateDistance, 1, test_array)

	pred_count=0
	nearest_k = np.apply_along_axis(sortArray, 1, distnaceArray)[:,:k]
	votes = map(str, np.apply_along_axis(vote, 1, nearest_k))
	# compare the predictions with the actual label
	for i in range(len(label)):
		if label.values()[i]==votes[i]:
			pred_count+=1
		# Write the prediction to the output file
		if i != len(label)-1:
			opfile.write(imageName[i] + " "+ votes[i]+"\n")
		else:
			opfile.write(imageName[i] + " "+ votes[i])
	accuracy =  pred_count * 100.0/len(votes)
	print "Accuracy of K Nearest neighbors where k:",k," :",accuracy
	
#### Ada boost Code Starts here ####
    
# Produce 1/0 label based on the classes for which the classifier is built
def label_estimator(label_data, label_value):
    return np.where(label_data == label_value, 1, 0) 


# Split the data based on classes
def train_data_split(data, label, class1, class2):
    row_idx = (np.argwhere((label == class1) | (label == class2))).ravel()
    split_data = data[row_idx,:]
    split_label = label_estimator(label[row_idx], class1)    
    return split_data, split_label


# Epsilon value: To calculate weighted error
def epsilon(sample_weight, hypothesis_pred, label_train):
    return np.sum(sample_weight*abs(hypothesis_pred-label_train))

# Re-weighting samples based on error from previous prediction
def sample_wt_estimator(stump_no, prev_weight, label_train, prev_hypothesis, N):
    if stump_no == 0:
        return np.where(label_train == 1, 1.0/(2*np.sum(label_train)), 1.0/(2*(N-np.sum(label_train))))
    else:
        epsilon_t = epsilon(prev_weight, prev_hypothesis, label_train)
        beta = epsilon_t/(1 - epsilon_t)
        weights = np.where(prev_hypothesis == label_train, prev_weight*beta, prev_weight)
        return weights/ np.sum(weights)

# Creating 20 stumps classifier based on the error:
def stump_creator(label_train, N, truth_dict):
    sample_weight = np.zeros(N)
    stumps = {}
    stump_keys = []
    tot_alpha = 0
    best_pred = np.zeros([N])
    for stump_no in range(15):
        sample_weight = sample_wt_estimator(stump_no, sample_weight, label_train, best_pred, N)
        error = 10000
        for key, pred in truth_dict.iteritems():
            if key not in stump_keys:
                [i,j] = key
                e = epsilon(sample_weight, pred, label_train)
                if e < error:
                    error = e
                    best_pred = pred
                    best_i = i
                    best_j = j
                    epsilon_val = epsilon(sample_weight, best_pred, label_train)
                    beta = epsilon_val/(1-epsilon_val)
                    alpha = np.log(1/beta)
        tot_alpha += alpha
        stumps[(best_i,best_j)] = alpha
        stump_keys = [key for key, value in stumps.iteritems()]
    return stumps

# Aggregating the steps required for training the model
def mainFunc(label_train,train_data):
    feature = train_data.transpose()
    N = len(label_train)  # N is the sample size
    K = len(feature) # K is the number of stumps
    truth_dict = {}
    for i in range(K):
        for j in range(K):
            if (i != j):
                truth_dict[(i,j)] = np.where(feature[i] >= feature[j], 1, 0)
    alpha_pred = stump_creator(label_train, N, truth_dict)
    return alpha_pred

# Predicts 1/-1 and multiplies by alpha for each classifier  
def prediction(dictionary, train_data):
    N = len(train_data)
    pred_output = np.zeros([N])
    for pair, alpha in dictionary.iteritems():
        i, j = pair
        list1 = train_data[:, i]
        list2 = train_data[:, j]
        output = np.where( list1 >= list2, alpha, -1*alpha )
        pred_output = np.add(pred_output,output)
    return pred_output

def dict_correcter(old_dict):
    new_dict = {}
    for key, value in old_dict.iteritems():
        a = key.encode('utf-8')
        new_key = tuple(map(lambda x: int(filter(str.isdigit,x)) ,a.split(',')))
        new_dict[(new_key)] = value
    return (new_dict)

def train_adaboost(file_name, model_file):
    # Reading the features and labels into separate numpy arrays 
    train_data_full = np.loadtxt(file_name, delimiter = ' ', usecols = range(2,194), dtype = int)
    train_label_full = np.loadtxt(file_name, delimiter = ' ', usecols = 1, dtype = str)
    
    # Splitting the input data and labels for each of the 6 classifiers 
    train_090, label_090 = train_data_split(train_data_full, train_label_full, '0', '90')
    train_0180, label_0180 = train_data_split(train_data_full, train_label_full, '0', '180')
    train_0270, label_0270 = train_data_split(train_data_full, train_label_full, '0', '270')
    train_90180, label_90180 = train_data_split(train_data_full, train_label_full, '90', '180')
    train_90270, label_90270 = train_data_split(train_data_full, train_label_full, '90', '270')
    train_180270, label_180270 = train_data_split(train_data_full, train_label_full, '180', '270')
    
    # Training 6 classifiers
    orientation_0_90 = mainFunc(label_090, train_090)
    orientation_0_180 = mainFunc(label_0180, train_0180)
    orientation_0_270 = mainFunc(label_0270, train_0270)
    orientation_90_180 = mainFunc(label_90180, train_90180)
    orientation_90_270 = mainFunc(label_90270, train_90270)
    orientation_180_270 = mainFunc(label_180270, train_180270)
    
    # Creating 6 other classifiers based on the previous 6:
    orientation_90_0    = {(key[1], key[0]): value for key , value in orientation_0_90.iteritems()}
    orientation_180_0   = {(key[1], key[0]): value for key , value in orientation_0_180.iteritems()}
    orientation_270_0   = {(key[1], key[0]): value for key , value in orientation_0_270.iteritems()}
    orientation_180_90  = {(key[1], key[0]): value for key , value in orientation_90_180.iteritems()}
    orientation_270_90  = {(key[1], key[0]): value for key , value in orientation_90_270.iteritems()}
    orientation_270_180 = {(key[1], key[0]): value for key , value in orientation_180_270.iteritems()}
    
    train_final_dict = {
    'orientation_0_90' : orientation_0_90,
    'orientation_0_180' : orientation_0_180,
    'orientation_0_270' : orientation_0_270,
    'orientation_90_0' : orientation_90_0,
    'orientation_90_180' : orientation_90_180,
    'orientation_90_270' : orientation_90_270,
    'orientation_180_0' : orientation_180_0,
    'orientation_180_90' : orientation_180_90,
    'orientation_180_270' : orientation_180_270,
    'orientation_270_0' : orientation_270_0,
    'orientation_270_90' : orientation_270_90,
    'orientation_270_180' : orientation_270_180,
    }
    # Writing model to output file:
    with open(model_file, 'w') as file:
        file.write(str(train_final_dict))

def test_adaboost(file_name, model_file):
    # Reading the features and labels into separate numpy arrays
    test_image_full = np.loadtxt(file_name, delimiter = ' ', usecols = 0, dtype = str)
    test_data_full = np.loadtxt(file_name, delimiter = ' ', usecols = range(2,194), dtype = int)
    test_label_full = np.loadtxt(file_name, delimiter = ' ', usecols = 1, dtype = str)
    overall_dict = eval(open(model_file, 'r').read())
    
    # Converting into separate dictionaries in the right format
    
    orientation_0_90 = overall_dict['orientation_0_90']
    orientation_0_180 = overall_dict['orientation_0_180']
    orientation_0_270 = overall_dict['orientation_0_270']
    orientation_90_0 = overall_dict['orientation_90_0']
    orientation_90_180 = overall_dict['orientation_90_180']
    orientation_90_270 = overall_dict['orientation_90_270']
    orientation_180_0 = overall_dict['orientation_180_0']
    orientation_180_90 = overall_dict['orientation_180_90']
    orientation_180_270 = overall_dict['orientation_180_270']
    orientation_270_0 = overall_dict['orientation_270_0']
    orientation_270_90 = overall_dict['orientation_270_90']
    orientation_270_180 = overall_dict['orientation_270_180']
    
    # Importing the model file and inserting stumps into dictionaries:
    ori090_test_pred = prediction(orientation_0_90, test_data_full)
    ori0180_test_pred = prediction(orientation_0_180, test_data_full)
    ori0270_test_pred = prediction(orientation_0_270, test_data_full)
    ori900_test_pred = prediction(orientation_90_0, test_data_full)
    ori90180_test_pred = prediction(orientation_90_180, test_data_full)
    ori90270_test_pred = prediction(orientation_90_270, test_data_full)
    ori1800_test_pred = prediction(orientation_180_0, test_data_full)
    ori18090_test_pred = prediction(orientation_180_90, test_data_full)
    ori180270_test_pred = prediction(orientation_180_270, test_data_full)
    ori2700_test_pred = prediction(orientation_270_0, test_data_full)
    ori27090_test_pred = prediction(orientation_270_90, test_data_full)
    ori270180_test_pred = prediction(orientation_270_180, test_data_full)
    
    ori0_test_pred   = (ori090_test_pred  + ori0180_test_pred  +   ori0270_test_pred)/3
    ori90_test_pred  = (ori900_test_pred  + ori90180_test_pred +  ori90270_test_pred)/3
    ori180_test_pred = (ori1800_test_pred + ori18090_test_pred + ori180270_test_pred)/3
    ori270_test_pred = (ori2700_test_pred + ori27090_test_pred + ori270180_test_pred)/3
    
    final_test_pred1 = np.empty([len(test_data_full)])
    for i in range(len(test_data_full)):
        pred_list = [ori0_test_pred[i], ori90_test_pred[i], ori180_test_pred[i], ori270_test_pred[i]]
        max_index = pred_list.index(max(pred_list))
        if max_index == 0:
            label = 0
        elif max_index == 1:
            label = 90
        elif max_index == 2:
            label = 180
        elif max_index == 3:
            label = 270
        final_test_pred1[i] = label 
    final_test_pred = np.array([str(int(i)) for i in final_test_pred1])
    print("Accuracy: ",np.sum(np.where(final_test_pred == test_label_full, 1, 0))*100.0/len(test_label_full),"%")
    final_output = pd.DataFrame({'image': test_image_full, 'prediction': final_test_pred})
    final_output.to_csv('output.txt', sep = ' ',index = False, header = False)

		
params=sys.argv
#type defines train/test
type=sys.argv[1]
#inputFile is either train/test file depending on the type
inputFile=sys.argv[2]
#Output file name of the model
modelFile=sys.argv[3]
#Model defines nnet/adaboost/knn or best
model=sys.argv[4]

#Training
if type=="train":
    if model=="nnet" or model=="best":
        weights,bias=trainNN(inputFile)
        nnet_model=np.array([weights,bias])
        np.save(modelFile,nnet_model)
    if model=="knn":
        trainKNN(inputFile,modelFile)
    if model=="adaboost":
        train_adaboost(inputFile, modelFile)
#Testing
else:
    if model=="nnet" or model=="best":
        nnet_model=np.load(modelFile)
        predict_nnet(inputFile,nnet_model)
    if model=="knn":
        testKNN(inputFile,modelFile)
    if model=="adaboost":
        test_adaboost(inputFile, modelFile)
