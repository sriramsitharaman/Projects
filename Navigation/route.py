#!/usr/bin/env python
# put your routing program here!

# Submission by: Shalaka Sharma, Vinita Boolchandani and Sriram Sitharaman
# Implemented - bfs, dfs, uniform, astar and cost functions => segments, distance, time, longtour

# (1) Which search algorithm seems to work best for each routing options?

# Algorithm: BFS
#   Costs: segments, distance and time
#       BFS is working well for short distances like for start and end cities
#       within the same state or neighboring states.
#       However the performance starts to deteriorate for longer distances.
#       It uses a queue as the fringe.
#   Costs: longtour
#       BFS performs poorly as it explores all the unnecessary paths
#       and doesn't put a preference of which successor it should explore first.
#
# Algorithm: DFS
#       Costs: segments, distance and time
#           DFS is also working well for short distances like for start and end cities within the same state.
#           However the performance starts to deteriorate for longer distances.
#           It spends time exploring from root to leaves when a promising solution
#           could have been found at just the 2nd level in the tree.
#           Performance of DFS is poorer than BFS. It uses a stack as the fringe.
#       Costs: longtour
#           DFS performs poorly as it explores all the unnecessary paths (depth-wise)
#           and doesn't put a preference of which successor it should explore first.
#
# Algorithm: Uniform
#   Costs: segments, distance and time, longtour
#       Uniform seems to be performing really well even though it doesn't use a heuristic.
#       It uses a Priority Queue as the fringe which does the magic and gets optimal solutions pretty fast.
#       Priority of a successor s is denoted by f(s) such that f(s) = g(s)
#       where g(s) is the actual cost from initial state to a successor s.
#       Now, the fringe pops out the state with the lowest priority and hence the computation time reduces drastically.
#       This helps in finding routes between cities far off from each other quickly.
#       Our code takes: 0.0914471149445 to compute an optimal path from Bridgeport,_New_Jersey to Alton,_California
#
# Algorithm: A*
#   Costs: segments, distance and time, longtour
#       A* seems to be performing really well owing to the heuristic function being used.
#       It uses a Priority Queue as the fringe which gets optimal solutions much faster.
#       Priority of a successor s is denoted by f(s) such that f(s) = g(s) + h(s)
#       where g(s) is the actual cost from initial state to a successor s
#       and h(s) is an admissible heuristic giving an approximation of cost
#       from successor s to the goal state (G).
#       Now, the fringe pops out the state with the lowest priority and hence the computation time reduces drastically.
#       This helps in finding routes between cities far off from each other quickly.
#       Our code takes: 0.11222410202 to compute an optimal path from Bridgeport,_New_Jersey to Alton,_California
#       which is marginally larger than Uniform.

# Thus, Uniform Cost Search works the best according to our code for all the four cost functions.
# This may be attributed to the heuristics used for A*.
# Although delay for A* is very less even for longer routes.
# Theoretically, A* should work best but the heuristic computation eats more time as compared to
# the implementation of Uniform Cost Search.
# So we conclude that Uniform Cost Search is working the best for us.

# ----------------------------------------------------------------------------------------------------------------------

# (2) Which algorithm is fastest in terms of the amount of computation time required by your
# program, and by how much, according to your experiments??

# Path: Virginia_Colony,_California to Los_Angeles,_California
#
#   Algorithm: DFS
#       Shortest path from Virginia_Colony,_California  to Los_Angeles,_California
#       Route: Virginia_Colony,_California Mission_Hills,_California Jct_I-5_&_CA_118,_California Panorama_City,_California North_Hollywood,_California Los_Angeles,_California
#       Total Distance: 48 miles
#       Total Time: 1.0803030303 hours
#       Avg. Computation Time: 0.00084400177002
#
#   Algorithm: BFS
#       Shortest path from Virginia_Colony,_California  to Los_Angeles,_California
#       Route: Virginia_Colony,_California Mission_Hills,_California Jct_I-5_&_CA_118,_California Panorama_City,_California North_Hollywood,_California Los_Angeles,_California
#       Total Distance: 48 miles
#       Total Time: 1.0803030303 hours
#       Avg. Computation Time: 0.000850915908813
#
#   Algorithm: Uniform
#       Shortest path from Virginia_Colony,_California  to Los_Angeles,_California
#       Route: Virginia_Colony,_California Mission_Hills,_California Jct_I-5_&_CA_118,_California Panorama_City,_California North_Hollywood,_California Los_Angeles,_California
#       Total Distance: 48 miles
#       Total Time: 1.0803030303 hours
#       Avg. Computation Time: 0.000252962112427
#
#   Algorithm: A*
#       Shortest path from Virginia_Colony,_California  to Los_Angeles,_California
#       Route: Virginia_Colony,_California Mission_Hills,_California Jct_I-5_&_CA_118,_California Panorama_City,_California North_Hollywood,_California Los_Angeles,_California
#       Total Distance: 48 miles
#       Total Time: 1.0803030303 hours
#       Avg. Computation Time: 0.000380039215088
#
# Path: Bridgeport,_New_Jersey  to Alton,_California
#
#   Algorithm: DFS
#       Avg. Computation Time: A LOT
#
#   Algorithm: BFS
#       Avg. Computation Time: A LOT
#
#   Algorithm: Uniform
#       Shortest path from Bridgeport,_New_Jersey  to Alton,_California
#       Total Distance: 2904 miles
#       Total Time: 53.9334887335 hours
#       Avg. Computation Time: 0.0853950977325
#
#   Algorithm: A*
#       Shortest path from Bridgeport,_New_Jersey  to Alton,_California
#       Total Distance: 2904 miles
#       Total Time: 53.9334887335 hours
#       Avg. Computation Time: 0.124378919601
#
# So, we conclude that Uniform runs the fastest ie. least computation time.

# ----------------------------------------------------------------------------------------------------------------------

# (3) Which algorithm requires the least memory, and by how much, according to your experiments?
#
# BFS consumes around (b*d) memory for fringe and takes around ~4.5 MB
# DFS consumes lesser memory than BFS as the branching factor b is much lesser than in BFS.
# Uniform and A* consume even memory as their space complexity is b^d ~25 MB
# We continue to search the whole tree and then printout optimal solution.
# As we don't stop after finding first solution as it may not be optimal, the memory consumption goes higher
# Also we store a lot of secondary information in the fringe to reduce computation time, hence our code uses more memory

# ----------------------------------------------------------------------------------------------------------------------

# (4) Which heuristic function(s) did you use, how good is it, and how might you make it/them better?

# First we decided to use Great Circle Distance to compute the distance in nautical miles between two cities
# While looking up formulas to compute Great Circle Distance, we came across a better Heuristic Function: GCD using Haversine Formula
# We then realized that it is over-estimating owing to errored data so we decided to use sqrt(Haversine Distance)
# This was again over-estimating and giving us sub-optimal paths in few cases. So we decided to use 4th root of Haversine distance
# This seemed to be working fine and gave us a good estimate for large number of inputs, so we decided to finalize it.
# We used this for cost_function = "distance" and cost_function = "longtour".
# For cost_function = "time", we divided the distance by 45, considering 45 miles/hour as the avg. speed
# For cost_function = "segments", we used heuristic function as 0. It is admissible and seems to be working well.
#
# To make it better:
#   We need to somehow handle missing information or wrong information in city-gps.txt file.
#   Junctions are not present in it and for this case we send heuristic as 0 which is not the best.
#   We could it add the data in the file or somehow compute a better heuristic value
#   by using prev estimated sitance and last length added
#   Also we could look for a consistent heuristic and use it

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


import sys
import heapq
import math

states = ['Mississippi', 'Iowa', 'Oklahoma', 'Wyoming', 'Minnesota', 'New_Jersey', 'Arkansas', 'Indiana', 'Maryland', 'Louisiana', 'New_Hampshire', 'Texas', 'New_York', 'Arizona', 'Wisconsin', 'Michigan', 'Kansas', 'Utah', 'Virginia', 'Oregon', 'Connecticut', 'Montana', 'California', 'Idaho', 'New_Mexico', 'South_Dakota', 'Massachusetts', 'Vermont', 'Georgia', 'Pennsylvania', 'Florida', 'North_Dakota', 'Tennessee', 'Nebraska', 'Kentucky', 'Missouri', 'Ohio', 'Alabama', 'Illinois', 'Colorado', 'Washington', 'West_Virginia', 'South_Carolina', 'Rhode_Island', 'North_Carolina', 'Nevada', 'Delaware', 'Maine']


class PriorityQueue:

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index,item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[2:]

    def __init__(self):
        self._queue = []
        self._index = 0


def convert_gps_to_dict(city_gps_file):
    gps = dict()
    for line in city_gps_file:
        values = line.split(' ')
        city = values[0]

        invalid_data = False

        try:
            lati = float(values[1])
        except ValueError:
            invalid_data = True
        try:
            longi = float(values[1])
        except ValueError:
            invalid_data = True

        city_data = dict()
        if not invalid_data:
            city_data['city'] = city
            city_data['lati'] = lati
            city_data['longi'] = longi
            if city not in gps:
                gps[city] = city_data

    return gps


def graphify_road_segments(road_segment_file):
    graph = dict()
    for line in road_segment_file:
        values = line.split(' ')
        start_city = values[0]
        end_city = values[1]

        try:
            length = int(values[2])
        except ValueError:
            length = 5
        if length == 0:
            length = 5
        try:
            speed_limit = int(values[3])
        except ValueError:
            speed_limit = 40
        if speed_limit == 0:
            speed_limit = 30
        highway = values[4].replace("\n","",1)

        adjacent_city = dict()
        adjacent_city['city'] = start_city
        adjacent_city['length'] = length
        adjacent_city['speed'] = speed_limit
        adjacent_city['highway'] = highway

        if end_city not in graph:
            graph[end_city] = [adjacent_city]
        else:
            graph[end_city].append(adjacent_city);

        adjacent_city = dict()
        adjacent_city['city'] = end_city
        adjacent_city['length'] = length
        adjacent_city['speed'] = speed_limit
        adjacent_city['highway'] = highway

        if start_city not in graph:
            graph[start_city] = [adjacent_city]
        else:
            graph[start_city].append(adjacent_city)

    return graph


def find_best_route(start_city, end_city, algo, cost):
    if algo == 'bfs':
        return bfs(cost, start_city, end_city)
    elif algo == 'dfs':
        return dfs(cost, start_city, end_city)
    elif algo == 'uniform':
        return uniform(cost, start_city, end_city)
    else:
        return astar(cost, start_city, end_city)


def get_successors(city, graph):
    return graph[city]


def is_goal(city, end_city):
    if city == end_city:
        return True
    return False

def get_cost(cost, s):
    if cost == 'segments':
        return 1
    elif cost == 'distance':
        return s['length']
    elif cost == 'time':
        return ((float)(s['length'])) / s['speed']
    elif cost == 'longtour':
        return -s['length']
    else: # cost == 'statetour'
        return s['length']


# http://introcs.cs.princeton.edu/python/12types/
def find_haversine_distance(start_city, end_city, cost):

    if cost == "distance" or cost == 'time':
        if start_city in gps and end_city in gps:
            start_city_values = gps[start_city]
            end_city_values = gps[end_city]

            start_city_lat = math.radians(start_city_values['lati'])
            start_city_long = math.radians(start_city_values['longi'])
            end_city_lat = math.radians(end_city_values['lati'])
            end_city_long = math.radians(end_city_values['longi'])

            angle1 = math.acos(math.sin(start_city_lat) * math.sin(end_city_lat) \
                               + math.cos(start_city_lat) * math.cos(end_city_lat) * math.cos(
                start_city_long - end_city_long))

            angle1 = math.degrees(angle1)

            distance1 = 60.0 * angle1
            if cost == 'time':
                return float(distance1 ** 0.25)/45
            return float(distance1 ** 0.25)
    return 0


def bfs(cost, start_city, end_city):
    if cost == "longtour":
        return bfs_longtour(cost, start_city, end_city)
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = [path]
    solution = dict()
    solution['route'] = []
    solution['cost'] = 500000
    visited_dict = dict()
    while len(fringe) > 0:
        popped_path = fringe.pop(0)
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        space_seperated_route = ' '.join(city for city in route)
        if cost_so_far > solution['cost']:
            continue
        if space_seperated_route not in visited_dict:
            visited_dict[space_seperated_route] = 1
        for s in get_successors(current_city, graph):
            new_path = dict()
            new_path['route'] = list(route)
            if s['city'] not in new_path['route']:
                new_path['route'].append(s['city'])
            else:
                continue
            new_path['cost'] = cost_so_far + get_cost(cost, s)
            if new_path['cost'] > solution['cost']:
                continue
            new_path['time'] = time_so_far + (float)(s['length']) / s['speed']
            new_path['distance'] = distance_so_far + s['length']
            new_path['successors'] = list(successors_list)
            new_path['successors'].append(s)
            if is_goal(s['city'], end_city):
                if solution['cost'] > new_path['cost']:
                    solution = new_path
            else:
                space_seperated_route_2 = ' '.join(city for city in new_path['route'])
                if space_seperated_route_2 not in visited_dict or new_path not in fringe:
                    fringe.append(new_path)
    return solution


def bfs_longtour(cost, start_city, end_city):
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = [path]
    solution = dict()
    solution['route'] = []
    solution['cost'] = 0
    visited_dict = dict()
    while len(fringe) > 0:
        popped_path = fringe.pop(0)
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        space_seperated_route = ' '.join(city for city in route)
        if space_seperated_route not in visited_dict:
            visited_dict[space_seperated_route] = 1
        for s in get_successors(current_city, graph):
            new_path = dict()
            new_path['route'] = list(route)
            if s['city'] not in new_path['route']:
                new_path['route'].append(s['city'])
            else:
                continue
            new_path['cost'] = cost_so_far + get_cost(cost, s)
            new_path['time'] = time_so_far + (float)(s['length']) / s['speed']
            new_path['distance'] = distance_so_far + s['length']
            new_path['successors'] = list(successors_list)
            new_path['successors'].append(s)
            if is_goal(s['city'], end_city):
                if solution['cost'] < new_path['cost']:
                    solution = new_path
            else:
                space_seperated_route_2 = ' '.join(city for city in new_path['route'])
                if space_seperated_route_2 not in visited_dict or new_path not in fringe:
                    fringe.append(new_path)
    return solution


def dfs(cost, start_city, end_city):
    if cost == "longtour":
        return dfs_longtour(cost, start_city, end_city)
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = [path]
    solution = dict()
    solution['route'] = []
    solution['cost'] = 500000
    while len(fringe) > 0:
        popped_path = fringe.pop()
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        for s in get_successors(current_city, graph):
            new_path = dict()
            new_path['route'] = list(route)
            if s['city'] not in new_path['route']:
                new_path['route'].append(s['city'])
            else:
                continue
            new_path['cost'] = cost_so_far + get_cost(cost, s)
            new_path['time'] = time_so_far + (float)(s['length'])/s['speed']
            new_path['distance'] = distance_so_far + s['length']
            new_path['successors'] = list(successors_list)
            new_path['successors'].append(s)
            if new_path['cost'] > solution['cost']:
                continue
            if is_goal(s['city'], end_city):
                if solution['cost'] > new_path['cost']:
                    solution = new_path
            else:
                fringe.append(new_path)
    return solution


def dfs_longtour(cost, start_city, end_city):
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = [path]
    solution = dict()
    solution['route'] = []
    solution['cost'] = 0
    while len(fringe) > 0:
        popped_path = fringe.pop()
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        for s in get_successors(current_city, graph):
            new_path = dict()
            new_path['route'] = list(route)
            if s['city'] not in new_path['route']:
                new_path['route'].append(s['city'])
            else:
                continue
            new_path['cost'] = cost_so_far + get_cost(cost, s)
            new_path['time'] = time_so_far + (float)(s['length'])/s['speed']
            new_path['distance'] = distance_so_far + s['length']
            new_path['successors'] = list(successors_list)
            new_path['successors'].append(s)
            if is_goal(s['city'], end_city):
                if solution['cost'] < new_path['cost']:
                    solution = new_path
            else:
                fringe.append(new_path)
    return solution


def uniform(cost,start_city,end_city):
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = PriorityQueue()
    fringe_len = 1
    fringe.push(path, 0)
    visited_dict = dict()
    solution = dict()
    solution['route'] = []
    solution['cost'] = 500000
    while fringe_len > 0:
        popped_path_tuple = fringe.pop()
        popped_path = popped_path_tuple[0]
        fringe_len -= 1
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        if is_goal(current_city,end_city):
            if solution['cost'] > cost_so_far:
                solution = popped_path
                break
        else:
            if current_city not in visited_dict:
                visited_dict[current_city] = 1
            else:
                continue
            for s in get_successors(current_city, graph):
                new_path = dict()
                new_path['route'] = list(route)
                new_path['route'].append(s['city'])
                new_path['cost'] = cost_so_far + get_cost(cost, s)
                new_path['time'] = time_so_far + (float)(s['length']) / s['speed']
                new_path['distance'] = distance_so_far + s['length']
                new_path['successors'] = list(successors_list)
                new_path['successors'].append(s)
                if s['city'] not in visited_dict:
                    fringe.push(new_path, new_path['cost'])
                    fringe_len += 1
    return solution


def astar(cost, start_city, end_city):
    path = dict()
    path['route'] = [start_city]
    path['cost'] = 0
    path['time'] = 0
    path['distance'] = 0
    start_node = {'city': start_city, 'length': 0, 'speed': 0, 'highway': ''}
    path['successors'] = [start_node]
    if start_city == end_city:
        path['successors'].append(start_node)
        path['route'].append(start_city)
        return path
    fringe = PriorityQueue()
    fringe_len = 1
    heuristic = find_haversine_distance(start_city, end_city, cost)
    fringe.push(path, 0+heuristic)
    visited_dict = dict()
    solution = dict()
    solution['route'] = []
    solution['cost'] = 500000
    while fringe_len > 0:
        popped_path_tuple = fringe.pop()
        popped_path = popped_path_tuple[0]
        fringe_len -= 1
        route = popped_path['route']
        cost_so_far = popped_path['cost']
        time_so_far = popped_path['time']
        distance_so_far = popped_path['distance']
        successors_list = popped_path['successors']
        current_city = route[-1]
        if is_goal(current_city, end_city):
            if solution['cost'] > cost_so_far:
                solution = popped_path
                break
        else:
            if current_city not in visited_dict:
                visited_dict[current_city] = 1
            else:
                continue
            for s in get_successors(current_city, graph):
                new_path = dict()
                new_path['route'] = list(route)
                new_path['route'].append(s['city'])
                new_path['cost'] = cost_so_far + get_cost(cost, s)
                new_path['time'] = time_so_far + (float)(s['length']) / s['speed']
                new_path['distance'] = distance_so_far + s['length']
                new_path['successors'] = list(successors_list)
                new_path['successors'].append(s)
                if s['city'] not in visited_dict:
                    heuristic = find_haversine_distance(s['city'], end_city, cost)
                    fringe.push(new_path, new_path['cost'] + heuristic)
                    fringe_len += 1
    return solution


def print_human_readable_info(result, cost_func):
    successors = result['successors']
    start_node = successors[0]
    end_node = successors[-1]
    if cost_func == 'segments':
        print "Path with least number of turns from", start_node['city'], " to", end_node['city']
        print "Total Turns:", result['cost']
        print "Total Distance:", result['distance'], "miles"
        print "Total Time:", result['time'], "hours"
    elif cost_func == 'distance':
        print "Shortest path from", start_node['city'], " to", end_node['city']
        print "Total Distance:", result['cost'], "miles"
        print "Total Time:", result['time'], "hours"
    elif cost_func == 'time':
        print "Fastest path from", start_node['city'], " to", end_node['city']
        print "Total Distance:", result['distance'], "miles"
        print "Total Time:", result['cost'], "hours"
    elif cost_func == 'longtour':
        print "Longest path from", start_node['city'], " to", end_node['city']
        print "Total Distance:", abs(result['cost']), "miles"
        print "Total Time:", result['time'], "hours"
    elif cost_func == 'statetour':
        print "State Tour path from", start_node['city'], " to", end_node['city']
        print "Total Distance", result['cost'], "miles"
        print "Total Time:", result['time'], "hours"
    print "Route:"
    prev = ""
    for i in range(len(successors)):
        if i != 0:
            print "On highway", (successors[i]['highway']).replace("\\n",'',1), "travel for", successors[i]['length'], "miles from", prev, "to",  successors[i]['city'], ". (Time:", round((((float)(successors[i]['length']))/successors[i]['speed']),3), "hours, Speed Limit:", successors[i]['speed'], "miles/hour)"
        prev = successors[i]['city']


def print_machine_readable_info(result):
    print result['distance'], result['time'] , ' '.join(city for city in result['route']) #', '.join(city for city in successors['route'])


road_segment_file = open("road-segments.txt","r")

city_gps_file = open("city-gps.txt","r")

start_city = sys.argv[1]
end_city = sys.argv[2]
algorithm = sys.argv[3]
cost = sys.argv[4]

algos = ['bfs', 'dfs', 'uniform', 'astar']
cost_functions = ['segments', 'distance', 'time', 'longtour', 'statetour']

if algorithm not in algos:
    algorithm = 'bfs'
if cost not in cost_functions:
    cost = 'distance'

graph = graphify_road_segments(road_segment_file)

gps = convert_gps_to_dict(city_gps_file)
result = find_best_route(start_city, end_city, algorithm, cost)
print_human_readable_info(result,cost)
print_machine_readable_info(result)




