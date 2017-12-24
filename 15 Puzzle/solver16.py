#!/usr/bin/env python

# Created by Sriram Sitharaman, Oct 2017
# 15 Puzzle solver
# Given a initial board configuration, this script tries to identify the optimal path for the solved goal_board
# board, if one exists
# If the initial board is not solvable, the script returns "Given Initial Board is not Solvable!"
# Solvability of the board is done by checking if the provided board has a even parity(sum of permutations inversions for each numbered tile except 0+row position of '0' tile)
#----------------------------------------------- Abstraction ----------------------------------------------------------
# Intial state: 4*4 dimensional input board with 15 tiles and a empty tile_locations
#
# Succ(board): All possible successors of the board that is achieved by sliding a
#    single tile from one cell into an empty cell, in this variant, either one, two, or three tiles may be slid
#    left, right, up or down in a single move. 
#
# Goal state: solved 4*4 goal board with tiles from 1 to 15 in order followed by the empty tile(0) at last
#
# Algorithm: A* Search
#    - Implemeted using Priority queue with the priority being g(n)+h(n) where h(n) is the heuristic from the Max Pattern database
#    - Includes a seen dictionary with the key as a hashvalue of the board to check and ignore revisting the same boards
# Heuristic: Max Pattern database
#    Reference - Searching with Pattern Databases,Joseph C. Culberson and Jonathan Schaeffer (pg. 402 - 415)
#    - 4 Pattern databases were built parallel using multiprocessing (patternDBGenerator.py)
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
#
#----------------------------------------------------------------------------------------------------------------------
#
# Discussed the concept of pattern databases with Shahidhya Ramachandran
 
import numpy as np
from random import shuffle
import copy
import os,sys
import json
import heapq
import time as time

# Returns the maxium of the heuristic value among the 4 pattern databases 
def db_heuristic(board):
    value=[]
    for i in range(4):
        loc=locateTiles(board,keys[i])
        try:
            value.append(pattern_db[i][loc])
        except:
            value.append(0)
    return max(value)

# Returns a unique hash value for a board
def hash_value(board):
	list=[]
	for i in range(4):
		for j in range(4):
			list.append(str(board[i][j]))
	return hash(" ".join(list))

#located the tiles for the partial pattern in the board
def locateTiles(board,key):
    board=np.array(board).reshape(16)
    #print (board)
    locations={board[i]:i+1 for i in range(16)}
    loc_list=[]
    #print (locations)
    for i in key:
        loc_list.append(locations[i])
    return " ".join([str(i) for i in loc_list])

# Checks the parity of the board
def check_solvability(board):
    parity=0
    zero_row=0
    locations={board[i][j]:(i,j) for i in range(4) for j in range(4)}
    for i in range(15,0,-1):
        for j in range(1,i):
            first_tile_loc=locations[i]
            second_tile_loc=locations[j]
            if first_tile_loc[0]<second_tile_loc[0]:
                parity+=1
            elif (first_tile_loc[0]==second_tile_loc[0]) and first_tile_loc[1]<second_tile_loc[1]:
                parity+=1
    return (parity+locations[0][0]+1)%2==0

# Creates the possible successors for the 4*4 board by moving empty tile 1,2,3 moves in L,R,U,D
def successors(current):
    path=current[0]
    board=current[1]
    step_count=current[2]
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
    
        if zero_i-i>0:
            temp1[[zero_i-i,zero_i-i-1],zero_j]=temp1[[zero_i-i-1,zero_i-i],zero_j]
            res.append((temp1.tolist(),step_count+1,path+" "+"D"+str(i+1)+str(zero_j+1)))
        #Bottom
        if zero_i+i<3:
            temp2[[zero_i+i,zero_i+i+1],zero_j]=temp2[[zero_i+i+1,zero_i+i],zero_j]
            res.append((temp2.tolist(),step_count+1,path+" "+"U"+str(i+1)+str(zero_j+1)))
    
        #Left
        if zero_j-i>0:
            temp3[zero_i,[zero_j-i,zero_j-i-1]]= temp3[zero_i,[zero_j-i-1,zero_j-i]]
            res.append((temp3.tolist(),step_count+1,path+" "+"R"+str(i+1)+str(zero_i+1)))
            
        #Right
        if zero_j+i<3:
            temp4[zero_i,[zero_j+i,zero_j+i+1]]= temp4[zero_i,[zero_j+i+1,zero_j+i]]
            res.append((temp4.tolist(),step_count+1,path+" "+"L"+str(i+1)+str(zero_i+1)))      
    return res
def is_goal(board):
    return board==goal_board.tolist()
def solve(initial_board):
    fringe=Fringe()
    initial_priority=db_heuristic(initial_board)
    #initial_priority=max_manhattan(initial_board,goal_board)
    #print (initial_priority)
    fringe.push(initial_board,initial_priority,0,"")
    fringe_len=1
    step_count=0
    count=1
    seen_fringe={hash_value(initial_board):1}
    closed={}
    while fringe_len> 0:
        fringe_len-=1
        current_node=fringe.pop()
        for s in successors(current_node):
            if is_goal(s[0]):
                #print (count,"Nodes added to the fringe\n")
                return(s[0],s[1],s[2])
            dist=db_heuristic(s[0])
            hashVal=hash_value(s[0])
            if hashVal not in seen_fringe:
                fringe.push(s[0],dist,s[1],s[2]) 
                fringe_len+=1
                count+=1
                seen_fringe[hashVal]=1

#Priority Queue implentation of Fringe using heapq
class Fringe:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority,step_count,path):
        heapq.heappush(self._queue, (priority+step_count, self._index, path,item,step_count))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[2:]
		
pattern_db=[]
cwd=os.getcwd()
keys=[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,14,15]]
for i in range(4):
    #Creating the 4-4-4-4 keys pattern database if the files are not present
    if not (os.path.isfile("db"+str(i)+".txt")):
        os.system("python patternDBGenerator.py")
    with open(os.path.join(cwd,"db"+str(i)+".txt"),'r') as inf:
        pattern_db.append(eval(inf.read()))



goal_board=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])
input_file=sys.argv[1]

initial_board=np.loadtxt(input_file)
if check_solvability(initial_board):
    x=time.time()
    goal_board,count_steps,steps=solve(initial_board.tolist())
    print ("Initial Board:")
    print (initial_board)
    print ("\nGoal Board:")
    print (np.array(goal_board))
    print "\n was solved in ", time.time()-x, " seconds"
    print "\n and solved using the following ",len(steps[1:].split(" ")), "steps:\n\n",steps[1:]
else:
    print "Given Initial Board is not Solvable!"