# 15 Puzzle solver
------------------
Given a initial board configuration, this script tries to identify the optimal path for the solved goal_board
board, if one exists. If the initial board is not solvable, the script returns "Given Initial Board is not Solvable!Solvability of the board is done by checking if the provided board has a even parity(sum of permutations inversions for each numbered tile except 0+row position of '0' tile)

Abstraction
-----------

Intial state
------------
4*4 dimensional input board with 15 tiles and a empty tile_locations

Successors
-----------
All possible successors of the board that is achieved by sliding a single tile from one cell into an empty cell, in this variant, either one, two, or three tiles may be slid
   left, right, up or down in a single move. 

Goal state
----------
solved 4*4 goal board with tiles from 1 to 15 in order followed by the empty tile(0) at last

Algorithm
----------
   - A* Search
   - Implemented using Priority queue with the priority being g(n)+h(n) where h(n) is the heuristic from the Max Pattern database
   - Includes a seen dictionary with the key as a hashvalue of the board to check and ignore revisting the same boards

Heuristic
---------
   - Max Pattern database
   Reference - Searching with Pattern Databases,Joseph C. Culberson and Jonathan Schaeffer (pg. 402 - 415)
   - 4 Pattern databases were built parallel using multiprocessing (patternDBGenerator.py)
   - A partial pattern of size 4 was taken , For instance below is one of the partial pattern
   ```

       1 2 - -     

       5 6 - -     

       - - - -    

       - - - 0    
   ```
   - A backwards BFS is done on the partial pattern where the successors are the admissible next step boards as stated in the problem
   - A step cost of 1 is incremented for each level of the BFS tree
   - The Final database contains all possible permutations of the above pattern and the actual cost to reach the above initial partial pattern
   - Database is a dictionary with key as the position of [1,2,5,6] in the board and value being the actual cost to reach the above initial partial pattern
   - Similarly, Pattern databases were built for the patterns of [3,4,7,8], [9,10,13,14] and [11,12,14,15]
   - Given any puzzle configuration, the maximum heuristic value among these 4 pattern databases was returned