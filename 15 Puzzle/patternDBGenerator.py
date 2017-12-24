#!/usr/bin/env python
# Created by Sriram Sitharaman, Oct 2017
# Utility Script to create 4-4-4-4 Pattern Databases
# Reference- Searching with Pattern Databases,Joseph C. Culberson and Jonathan Schaeffer (pg. 402 - 415)
#    - 4 Pattern databases were built parallel using multiprocessing 
#    - A partial pattern of size 4 was taken , For instance below is one of the partial pattern
#
#    1 2 - - 
#    5 6 - - 
#    - - - -
#    - - - 0
#    
#    - A backwards BFS is done on the partial pattern where the successors are the admissible next step boards as stated in the problem
#    - A step cost of 1 is incremented for each level of the BFS tree
#    - The Final database contains all possible permutations of the above pattern and the actual cost to reach the above initial partial pattern
#    - Database is a dictionary with key as the position of [1,2,5,6] in the board and value being the actual cost to reach the above initial partial pattern
#    - Similarly, Pattern databases were built for the patterns of [3,4,7,8], [9,10,13,14] and [11,12,14,15]
#    - Given any puzzle configuration, the maximum heuristic value among these 4 pattern databases was returned

import multiprocessing as mp
import numpy as np
import copy
import time as time
import json

def hash_value(board):
	list=[]
	for i in range(4):
		for j in range(4):
			list.append(str(board[i][j]))
	return hash(" ".join(list))

def successors(current):
    board=current[0]
    step_count=current[1]
    tile_locations={board[i][j]:[i,j] for i in range(4) for j in range(4)}
    zero_loc=tile_locations[0]
    zero_i=zero_loc[0]
    zero_j=zero_loc[1]
    res=[]
    board=np.array(board)
    temp1=copy.deepcopy(board)
    temp2=copy.deepcopy(board)
    temp3=copy.deepcopy(board)
    temp4=copy.deepcopy(board)
    for i in range(3):
		#Top
        if zero_i-i>0:
            temp1[[zero_i-i,zero_i-i-1],zero_j]=temp1[[zero_i-i-1,zero_i-i],zero_j]
            res.append((temp1.tolist(),step_count+1))
        #Bottom
        if zero_i+i<3:
            temp2[[zero_i+i,zero_i+i+1],zero_j]=temp2[[zero_i+i+1,zero_i+i],zero_j]
            res.append((temp2.tolist(),step_count+1))
    
        #Left
        if zero_j-i>0:
            temp3[zero_i,[zero_j-i,zero_j-i-1]]= temp3[zero_i,[zero_j-i-1,zero_j-i]]
            res.append((temp3.tolist(),step_count+1))
            
        #Right
        if zero_j+i<3:
            temp4[zero_i,[zero_j+i,zero_j+i+1]]= temp4[zero_i,[zero_j+i+1,zero_j+i]]
            res.append((temp4.tolist(),step_count+1))      
    return res

def locateTiles(board,key):
	board=np.array(board).reshape(16)
	locations={board[i]:i+1 for i in range(16)}
	loc_list=[]
	for i in key:
		loc_list.append(locations[i])
	return " ".join([str(i) for i in loc_list])
	
def generateDB(initial_board,key,pos,PatternDBList):
	count=0
	seenfringe={hash_value(initial_board):1}
	fringe = [(initial_board,0)]
	fringeDict={locateTiles(initial_board,key): 0}
	while len(fringe)> 0:
		for s in successors(fringe.pop(0)):
			#print (s)
			hashVal=hash_value(s[0])
			if hashVal not in seenfringe:
				count+=1
				seenfringe[hashVal]=1
				fringe.append(s)
				tileLocations=locateTiles(s[0],key)
				if tileLocations not in fringeDict:
					fringeDict[tileLocations]=s[1]
	PatternDBList.put((pos,fringeDict))

if __name__ == '__main__':
	mp.freeze_support()
	PatternDBList=mp.Queue()
	board1=[[1,2,17,17],[5,6,17,17],[17,17,17,17],[17,17,17,0]]
	board2=[[17,17,3,4],[17,17,7,8],[17,17,17,17],[17,17,17,0]]
	board3=[[17,17,17,17],[17,17,17,17],[9,10,17,17],[13,14,17,0]]
	board4=[[17,17,17,17],[17,17,17,17],[17,17,11,12],[17,14,15,0]]
	boards=[board1,board2,board3,board4]
	keys=[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,14,15]]
	
	print "4-4-4-4 Pattern Databases generation has started"
	# Setup a list of processes for generating 4 pattern databases
	processes = [mp.Process(target=generateDB, args=(boards[no], keys[no],no,PatternDBList)) for no in range(4)]
	
	# Run the 4 processes for generating 4 pattern databases
	for process in processes:
		process.start()

	# Fetch the Pattern DataBases from the output queue
	out = [PatternDBList.get() for p in processes]
	for patternDB in out:
		#print (i[0],len(i[1]))
		with open('db'+str(patternDB[0])+'.txt', 'w') as outfile:
			json.dump(patternDB[1], outfile)
	
	#Exit the completed processes
	for process in processes:
		process.join()
	print "Pattern Databases has been generated"