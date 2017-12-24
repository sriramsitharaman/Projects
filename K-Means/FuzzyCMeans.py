from __future__ import division
from tkinter import filedialog
import numpy
from numpy.random import *

#Kmeans ++ Centroid Initialization
import random
def initializeCentriods(Data, K):
    #Taking the random data point as the first centroid using a uniformly random variable
    Centroids = [Data[random.randint(0,len(Data)-1)]]
    #Finding the next K-1 centroids using a probability distribution of distance between data point and the centroid
    #with minimum distance
    for _ in range(K-1):
        DistanceSq = numpy.array([min([numpy.inner(numpy.array(Centroid)-numpy.array(row),numpy.array(Centroid)-numpy.array(row))**0.5 for Centroid in Centroids]) for row in Data])
        individualProbabilities = DistanceSq/DistanceSq.sum()
        cumulativeprobabilities= individualProbabilities.cumsum()
        randVar = rand()
        FinalIndex=-1
        for index,probability in enumerate(cumulativeprobabilities):
            if randVar<probability:
                FinalIndex=index
                break
        Centroids.append(Data[numpy.argmax(individualProbabilities)])
    return Centroids

#Reading the content from the file and separting the attributes and the class variable C
def readfile(filename):
  lines=[line for line in open(filename)]
  lines.pop(0)
  rownames=[]
  data=[]
  classVar=[]
  for line in lines:
    p=line.strip().replace("?","0").split(',')
    rownames.append(p[0])
    data.append([float(x) for x in p[1:len(p)-1]])
    #data.append([float(x) for x in p])
    classVar.append(int(p[9]))
  return rownames,data,classVar

#For finding the Total Error rate of all the Clusters created
def totalErrorRate(CentriodIndex,classVar,K):
    CentError=[]
    TotalError=0
    #A Dictionary of count of benign and malignant patients is initialized along with
    #the type of the cluster and the error
    for i in range(K):
        CentError.append({'2':0,'4':0,'type':0,'error':0})
    #Count of benign and malignant patients is updated for each cluster
    for i in range(numRows):
        if classVar[i]==2:
            CentError[CentriodIndex[i]]['2'] +=1
        else:
            CentError[CentriodIndex[i]]['4'] +=1
    #Type and Error rate of each cluster is identified    
    for i in range(K):
        Total=CentError[i]['2']+CentError[i]['4']
        CentError[i]['type']= 2 if CentError[i]['2']>=CentError[i]['4'] else 4
        CentError[i]['error']= 1.00*CentError[i]['4']/Total if CentError[i]['2']>=CentError[i]['4'] else 1.00*CentError[i]['2']/Total
        TotalError+=CentError[i]['error']
    return CentError,TotalError

#For getting the file
Filename=filedialog.askopenfile(mode='r')
rows,data,classVar=readfile(Filename.name)
Path=File[:k+1]
numRows,numCols=numpy.array(data).shape

#Number of rows and columns in the data
print (numRows,numCols)

#Fuzzy CMeans Implementation
itermax=10
loopiter=0
BestError=9999999999999
BestCentroidIndex=[]
for tau in range(5):
    for K in range(2,6):
        CentroidMembership=[]
        Centroids = initializeCentriods(data,K)
        print (Centroids)
        loopiter=0
        while True:
            loopiter=loopiter+1
            
            for row in data: 
                CentroidDiff = [numpy.inner(numpy.array(centroid)-numpy.array(row),numpy.array(centroid)-numpy.array(row))**0.5 for centroid in Centroids]
                CentroidMem = [1/(x+1) for x in CentroidDiff]
                CentroidMembership.append(CentroidMem/sum(CentroidMem))
            newCentroids=numpy.empty([K, numCols])
            membershipCount=0.0
            for i in range(K):
                sumCentroid=numpy.array([float(0)]*numCols)
                for j in range(numRows):
                    sumCentroid+=[x*CentroidMembership[j][i] for x in data[j]]
                    membershipCount+=CentroidMembership[j][i]
                newCentroids[i]=sumCentroid/membershipCount
            thresh=Centroids-newCentroids
            taucheck=(1/K)*sum([(numpy.inner(x,x))**0.5 for x in thresh])
            CentriodIndex=[]
            if taucheck<=tau or loopiter>=itermax:
                for x in CentroidMembership:
                    maxMem=0
                    maxindex=0
                    for i in range(len(x)):
                        if x[i]>maxMem:
                            maxMem=x[i]
                            maxindex=i
                    CentriodIndex.append(maxindex)
                CentError,error=totalErrorRate(CentriodIndex,classVar,K)
                if BestError>error:
                    BestError=error
                    BestCentroidIndex=CentriodIndex
                    BestK=K
                    BestTau=tau
                    BestCentError=CentError
                    BestCentroids=newCentroids
                print ("K : ",K, "Convering Iteration : ", loopiter ,"Tau : ", tau, "Threshold : " ,taucheck, "Error : ",error)
                break
            else:
                Centroids=newCentroids