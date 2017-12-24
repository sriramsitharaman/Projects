#!/usr/bin/env python
import numpy as np
import heapq as hq
import sys

##################################################      Methodology    ################################################################
# Creates the next best step using minimax and alpha beta pruning with a heuristic function for evaluation of the board`
# An iterative deeping scheme has been employed to search from a depth of 1 till 6
# To make alpha beta pruning effective, ordering of successor boards (moves) has been employed using the below techniques
#
###################
#Successor Function
###################
#
#  For each piece type in the board (kingfisher,quetzal,bishop,nighthawk,parakeet and robin), an utility function is defined to 
# identify the all possible and admissible positions to which the respective piece can move
#
###################
#Evaluation function for a given pichu board
###################
#
#Identifies the weighted difference between piece counts of current player and opponent player
# Worth of the pieces are obtained from the below page
# http://chessprogramming.wikispaces.com/Simplified+evaluation+function
#
#
###############
# Move Ordering
###############
# The successors are ordered based on the below three move ordering methods (in increasing order of priority)
# 1) A global dictionary called pv stack (prinicipal variation) stores the path containing all boards that lead to first beta cut off
# 2) A heuristic dictionary containing the heuristic value of board is also maintained. From this we can identify if a particular move is a capture
#    or a non-capture move
# 3) Also, all the moves that lead to a beta cut off anywhere in the search is maintained in a dictionary called killer_moves
#
#
# Ordering of the moves has been done so that while looking at the next depth in an iterative deeping scheme makes the search faster
# A terminal condition is achieved when there is only one kingfisher in the board
# The algorithm also stop searching further depth in the search tree if a depth of maximum depth is attained
#########################################################################################################################################


killer_moves={} #Stores all the moves that resulted in a beta cut off 
global_pv_path_nodes=pv_path_nodes={} #Stores the path (from root to leaf) of the moves that resulted in the first beta cut off  
pv_flag=0 # To check if the first beta cut off has been achieved
heuristic_dict={} # Contains the evaluation value of a pichu board 


# Evaluation function for a given pichu board
# Identifies the weighted difference between piece counts of current player and opponent player
# Worth of the pieces are obtained from the below page
# http://chessprogramming.wikispaces.com/Simplified+evaluation+function
def evaluation(board):
    birds_dict=get_dict(board)
    opponent_player="b" if START_PLAYER=="w" else "w"
    white_list=['K','Q','R','B','N','P'] 
    black_list=['k','q','r','b','n','p']
    worth=[20000,900,500,330,320,100]
    materialscore=0
    piecesq_score=0
    current_player_pieces=white_list if START_PLAYER=="w" else black_list
    opponent_player_pieces=black_list if START_PLAYER=="w" else white_list
    for i in range(6):
        piece=current_player_pieces[i]
        try:
            current_pos=birds_dict[piece]
            current_cnt=len(current_pos)
        except:
            current_pos=[]
            current_cnt=0
        oppo_piece=opponent_player_pieces[i]
        try:
            oppo_pos=birds_dict[oppo_piece]
            oppo_cnt=len(oppo_pos)
        except:
            oppo_pos=[]
            oppo_cnt=0
        materialscore+=worth[i]*(current_cnt-oppo_cnt)
    score=materialscore
    hashval=hash_val(board)
    if hashval not in heuristic_dict:
        heuristic_dict[hashval]=score
    return score

#returns the string of the board
def hash_val(board):
    return str(board)

#For identifying the order in which a particular move has to be considered from the given successors of a parent board
#The successors are ordered based on the below three move ordering methods (in increasing order of priority)
# 1) A global dictionary called pv stack (prinicipal variation) stores the path containing all boards that lead to first beta cut off
# 2) A heuristic dictionary containing the heuristic value of board is also maintained. From this we can identify if a particular move is a capture
#    or a non-capture move
# 3) Also, all the moves that lead to a beta cut off anywhere in the search is maintained in a dictionary called killer_moves
def find_priority(succ,current_player):
    hashval=hash_val(succ)
    if hashval in global_pv_path_nodes:
        return -10000000
    try:
        return -1*heuristic_dict[hashval]
    except:
        pass
    if hashval in killer_moves:
        return -5000 
    return 0

#Returns the dictionary containing list of all positions unique pieces in the board	
def get_dict(board):
    birds_dict = {}
    for i in range(8):
        for j in range(8):
            if board[i][j]!='.':
                if board[i][j] not in birds_dict:
                    birds_dict[board[i][j]]=[(i,j)]
                else:
                    birds_dict[board[i][j]].append((i,j))
    return birds_dict

#Moves the piece from source location to target location
def move(board,i1,j1,i2,j2,current_player):
    choose_q='Q' if current_player=='w' else 'q'
    temp_board=[[j for j in i] for i in board]
    current_piece = board[i1][j1]
    temp_board[i2][j2] = choose_q if i2%7==0 and current_piece in ['p','P'] else current_piece
    temp_board[i1][j1] = '.'
    return temp_board

# Finds the all admissible positions to which robin can be moved
def find_legal_robin_positions(board,i1,j1,attackList):
    #top
    for step in range(1,8):
        if i1-step>=0 and (board[i1-step][j1]=='.'):
            yield (i1-step,j1)
        elif  i1-step>=0 and  board[i1-step][j1] in attackList:
            yield (i1-step,j1)
            break
        else:
            break
    #bottom
    for step in range(1,8):
        if i1+step<=7 and (board[i1+step][j1]=='.'):
            yield (i1+step,j1)
        elif  i1+step<=7 and  board[i1+step][j1] in attackList:
            yield (i1+step,j1)
            break
        else:
            break
    #left
    for step in range(1,8):
        if j1-step>=0 and (board[i1][j1-step]=='.'):
            yield (i1,j1-step)
        elif  j1-step>=0 and  board[i1][j1-step] in attackList:
            yield (i1,j1-step)
            break
        else:
            break
    #right
    for step in range(1,8):
        if j1+step<=7 and (board[i1][j1+step]=='.'):
            yield (i1,j1+step)
        elif  j1+step<=7  and  board[i1][j1+step] in attackList:
            yield (i1,j1+step)
            break
        else:
            break

# Finds the all admissible positions to which bluejay can be moved
def find_legal_bluejay_positions(board,i1,j1,attackList):
    #top left diagonal
    for step in range(1,8):
        if i1-step>=0 and j1-step>=0 and (board[i1-step][j1-step]=='.'):
            yield (i1-step,j1-step)
        elif i1-step>=0 and j1-step>=0 and board[i1-step][j1-step] in attackList:
            yield (i1-step,j1-step)
            break
        else:
            break
    #bottom left diagonal
    for step in range(1,8):
        if i1+step<=7 and j1-step>=0 and (board[i1+step][j1-step]=='.'):
            yield (i1+step,j1-step)
        elif  i1+step<=7 and j1-step>=0 and board[i1+step][j1-step] in attackList:
            yield (i1+step,j1-step)
            break
        else:
            break
    #top right diagonal
    for step in range(1,8):
        if i1-step>=0 and j1+step<=7 and (board[i1-step][j1+step]=='.'):
            yield (i1-step,j1+step)
        elif  i1-step>=0 and j1+step<=7 and board[i1-step][j1+step] in attackList:
            yield (i1-step,j1+step)
            break
        else:
            break
    #bottom right diagonal
    for step in range(1,8):
        if i1+step<=7 and j1+step<=7 and (board[i1+step][j1+step]=='.'):
            yield (i1+step,j1+step)
        elif  i1+step<=7 and j1+step<=7  and board[i1+step][j1+step] in attackList:
            yield (i1+step,j1+step)
            break
        else:
            break

# Finds the all admissible positions to which nighthawk can be moved
def find_legal_nighthawk_positions(board,i1,j1,attackList):
    positions=[(i1+1,j1-2),(i1+2,j1-1),(i1+1,j1+2),(i1+2,j1+1),(i1-1,j1-2),(i1-2,j1-1),(i1-1,j1+2),(i1-2,j1+1)]
    for pos in positions:
        i2,j2=pos
        if ((i2>=0 and i2<=7) and (j2>=0 and j2<=7)) and (board[i2][j2]=='.' or board[i2][j2] in attackList):
            yield (i2,j2)

# Finds the all admissible positions to which KingFisher can be moved
def find_legal_kingfisher_positions(board,i1,j1,attackList):
    positions=[(i1+1,j1),(i1-1,j1),(i1,j1-1),(i1,j1+1),(i1-1,j1-1),(i1-1,j1+1),(i1+1,j1-1),(i1+1,j1+1)]
    for pos in positions:
        i2,j2=pos
        if ((i2>=0 and i2<=7) and (j2>=0 and j2<=7)) and (board[i2][j2]=='.' or board[i2][j2] in attackList):
            yield (i2,j2)

# Finds the all admissible positions to which Parakeet can be moved
def find_legal_parakeet_positions(board,i1,j1,attackList,current_player):
    white_moves=[(i1+1,j1),(i1+1,j1+1),(i1+1,j1-1),(i1+2,j1)]
    black_moves=[(i1-1,j1),(i1-1,j1-1),(i1-1,j1+1),(i1-2,j1)]
    positions=white_moves if current_player=='w' else black_moves
    for pos in positions[1:-1]:
        i2,j2=pos
        if ((i2>=0 and i2<=7) and (j2>=0 and j2<=7)) and board[i2][j2] in attackList:
            yield (i2,j2)
    i2,j2=positions[0]
    if ((i2>=0 and i2<=7) and (j2>=0 and j2<=7)) and board[i2][j2]=='.':
        yield (i2,j2)
    i3,j3=positions[3]
    if (i1%6 if current_player!='w' else i1-1)==0 and board[i2][j2]=='.' and board[i3][j3]=='.':
        yield positions[3]

#Identifies all the possible successors for a given board 	
def get_successors(board,current_player):
    #board=np.array(board)
    birds_dict = get_dict(board)
    ##print (birds_dict)
    successors=[]
    direction=-1 if current_player=="w" else 1
    attackList={'P':1,'B':1,'Q':1,'K':1,'R':1,'N':1} if current_player!="w" else {'p':1,'b':1,'q':1,'k':1,'r':1,'n':1}

    try:
        choose='Q' if current_player=="w" else 'q'
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_bluejay_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
            for pos in find_legal_robin_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass

    try:
        choose='R' if current_player=="w" else 'r'
        #Robin
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_robin_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass
    try:
        choose='B' if current_player=="w" else 'b'
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_bluejay_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass
    try:
        choose='N' if current_player=="w" else 'n'
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_nighthawk_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass

    try:
        choose='P' if current_player=="w" else 'p'
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_parakeet_positions(board,i1,j1,attackList,current_player):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass
    try:
        choose='K' if current_player=="w" else 'k'
        for par in birds_dict[choose]:
            i1,j1=par
            for pos in find_legal_kingfisher_positions(board,i1,j1,attackList):
                curr_succ=move(board,i1,j1,pos[0],pos[1],current_player)
                hq.heappush(successors,(find_priority(curr_succ,current_player),curr_succ))
    except:
        pass
    return successors

# Check if the board is end board or not
# (i.e.) returns the difference of Kingfisher counts of our piece and opponent piece
def isTerminal(board,current_player):
    board_str=str(board)
    opponent_player="b" if current_player=="w" else "w"
    our_king='K' if current_player=="w" else "k"
    opp_king='k' if current_player=="w" else "K"
    our_count=0
    opp_count=0
    if our_king in board_str:
        our_count+=1
    if opp_king in board_str:
        opp_count+=1
    return our_count-opp_count

###############Minimax with Alpha-Beta pruning#################

#Max function called with a board , maximum depth , current depth, alpha value, beta value and the respective player
# PV stack is updated based on if the first beta cut off has been achieved or not
# Killer moves is updated for all the beta cut off boards	

def max_play(board,max_depth,depth,alpha,beta,current_player):
    global pv_flag
    global pv_path_nodes

    if depth == max_depth or isTerminal(board,current_player)!=0:
        if pv_flag!=1:
            del pv_path_nodes[depth]
        return evaluation(board)
    for move in get_successors(board,current_player):
        if pv_flag==0:
            pv_path_nodes[depth+1]=hash_val(move[1])
        alpha = max(alpha, min_play(move[1],max_depth,depth+1, alpha,beta,"b" if current_player=="w" else "w"))
        if alpha >= beta:
            pv_path_nodes[depth+1]=hash_val(move[1])
            pv_flag=1
            killer_moves[hash_val(board)]=depth-1
            return alpha
    return alpha

#Min function called with a board , maximum depth , current depth, alpha value, beta value and the respective player
# PV stack is updated based on if the first beta cut off has been achieved or not
# Killer moves is updated for all the beta cut off boards	
def min_play(board,max_depth,depth,alpha,beta,current_player):
    global pv_flag
    global pv_path_nodes
    if depth == max_depth or isTerminal(board,current_player)!=0:
        if pv_flag!=1:
            del pv_path_nodes[depth]
        return evaluation(board)
    for move in get_successors(board,current_player):
        if pv_flag==0:
            pv_path_nodes[depth+1]=hash_val(move[1])
        beta = min(beta, max_play(move[1],max_depth,depth+1, alpha,beta,"b" if current_player=="w" else "w"))
        if alpha >= beta:
            pv_path_nodes[depth+1]=hash_val(move[1])
            pv_flag=1
            killer_moves[hash_val(board)]=depth-1
            return beta
    return beta

# Minimax algorithm which calls the min function for all the successor boards of the starting player 
# if the length of the pv_path_nodes is not the max_depth, then it means that beta cut off is not achieved at
# that particular successor, hence the dictionary is re initialized, otherwise pv_flag is set to 1 which
# ensures the pv_path_nodes is not updated further
def minimax(board,current_player,max_depth,depth):
    global pv_flag
    global pv_path_nodes

    alpha = -10000000
    beta = 10000000
    next_moves = get_successors(board,current_player)
    check=[]
    for move in next_moves:
        if pv_flag==0:
            pv_path_nodes[depth+1]=hash_val(move[1])
        check.append(min_play(move[1],max_depth,depth+1,alpha,beta,"b" if current_player=="w" else "w"))
        if len(pv_path_nodes)!=max_depth:
            pv_path_nodes={}

    result=np.array(next_moves[check.index(max(check))][1])

    return "".join([result[i][j] for i in range(8) for j in range(8)])
    


board=sys.argv[2]
START_PLAYER=sys.argv[1]
time=sys.argv[3]

board=np.array(list(board)).reshape(8,8)


#Iterative Deepening from a depth of 1 to 5
for i in range(1,6):
    global_pv_path_nodes=pv_path_nodes
    pv_path_nodes={}
    pv_flag=0
    next_board=minimax(board.tolist(),START_PLAYER,i,0)
    if isTerminal(next_board,START_PLAYER)!=0:
        print (next_board)
        break
    print (next_board)
