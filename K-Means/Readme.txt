

Procedure for running the code:

Programming Language used/needed: Python 3
Packages needed to run the code : tkinter,numpy,random

Steps to run the code:

1) Once the packages dependencies are installed,run the Kmeans.py code
2) A Tkinter dialog box would open asking for the input data file
3) Two csv files have been included in the submission folder
	DeltaFix.csv - After missing values are fixed
	DataClean2.csv - After highly correlated files are removed
4) Once the desired data is loaded, the program would start running
5) An output file called KmeansOutput.csv with the error rates for 20 iteration and K=2 to 5 would be generated in the same location as your input data file
6) This output file was used to generate the error rate plot using R code. (Also,the run of Kmeans which used to create the included errro plot is also included termed as "KmeansOutput-Rplot.csv")