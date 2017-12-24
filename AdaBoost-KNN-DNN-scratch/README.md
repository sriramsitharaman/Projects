# a4

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
