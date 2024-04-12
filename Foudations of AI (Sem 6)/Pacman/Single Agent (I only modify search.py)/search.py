# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from queue import Queue, PriorityQueue
from copy import deepcopy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    #initializing stuff
    visited = set()
    q = Queue()
    path = []

    cur_pos = maze.getStart()
    q.put(cur_pos)
    visited.add(cur_pos)

    #to keep track of parent nodes
    dim = maze.getDimensions()
    #dim[0] is the height, dim[1] is the width
    parents = [[None for _ in range(0, dim[1]+1)] for _ in range(0, dim[0]+1)]
    parents[cur_pos[0]][ cur_pos[1]] = (-1,-1)

    #bfs 
    while not q.empty(): 
        cur_pos = q.get()
        
        if maze.isObjective(cur_pos[0], cur_pos[1]):
            goal_pos = cur_pos
            while cur_pos != (-1,-1):
                path.append(cur_pos)
                cur_pos = parents[cur_pos[0]][cur_pos[1]]
            
            path.reverse()
            return path

        neighbors = maze.getNeighbors(cur_pos[0], cur_pos[1])

        for i in neighbors:
            if i not in visited:
                parents[i[0]][i[1]] = cur_pos
                visited.add(i)
                q.put(i)

    

def manhattan_distance(start ,goal):
    return (abs(goal[0] - start[0]) + abs(goal[1]-start[1]))

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    #initializing stuff

    start = maze.getStart()
    obj = maze.getObjectives()[0]
    
    open_list = {start: manhattan_distance(start, obj)}
    closed_list = dict()

    parents = {start: (-1,-1)}
    gn = {start: 0}

    path = []

    while len(open_list):
        #referring to gn + hn
        min = 1000000
        for i in open_list:
            if open_list[i] < min:
                min= open_list[i]
                cur = i

        closed_list.update({cur:min})
        del open_list[cur]

        if cur == obj:
            #do the path here
            while cur != (-1,-1):
                path.append(cur)
                cur = parents[cur]
            
            path.reverse()
            return path
        
        neighbors = maze.getNeighbors(cur[0], cur[1])
        for i in neighbors:
            gn_neighbors = gn[cur]+1
            if i in open_list:
                if gn_neighbors < gn[i]:
                    parents.update({i: cur})
                    open_list.update({i: gn_neighbors + manhattan_distance(i,obj)})
            elif i in closed_list:
                if gn_neighbors < gn[i]:
                    #migrate from closed list to open list
                    parents.update({i: cur})
                    open_list.update({i: gn_neighbors + manhattan_distance(i,obj)})
                    del closed_list[i]
            else:
                gn.update({i: gn_neighbors})
                parents.update({i: cur})
                #gn+hn
                open_list.update({i: gn_neighbors + manhattan_distance(i,obj)})

        

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return (astar_multi(maze))


def actual_cost_goals(objectives, maze_copy, dist_goals):
    obj_len = len(objectives)
    #get the actual cost from i goal to j goal
    for i in range(obj_len):
        for j in range(i+1, obj_len):
            #set i goal as the start
            maze_copy.setStart(objectives[i])
            #set j goal as the objective
            maze_copy.setObjectives([objectives[j]])
            path_len = len(astar(maze_copy))-1

            #store the path len into a dic
            dist_goals.update({(objectives[i],objectives[j]): path_len})
            dist_goals.update({(objectives[j],objectives[i]): path_len })

def neighbors_goals(objectives, pos):
    ans = []
    for i in objectives:
        if pos != i:
            ans.append(i)

    return tuple(ans)

def heuristic(cur_pos, objectives, dist_goals):
    not_yet = list(objectives)
    q2= PriorityQueue()

    obj_len = len(objectives)
    if obj_len == 0:
        return 0
    cur = objectives[0]

    mst = 0
    while True:
        obj_len -= 1
        if obj_len == 0:
            break

        not_yet.remove(cur)
        for i in not_yet:
            q2.put((dist_goals[cur,i], i))

        while True:
            min_edge = q2.get()
            cur = min_edge[1]
            if cur in not_yet:
                break

        mst += min_edge[0]
        

    #manhattan distance
    min = 1000000
    for i in objectives:
        manhattan_distance = abs(i[1] - cur_pos[1]) + abs(i[0] - cur_pos[0])
        if manhattan_distance < min:
            min = manhattan_distance

    return (mst + min)



def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    #(i,j) = distance between i and j
    dist_goals = dict()
    objectives = maze.getObjectives()
    maze_copy = deepcopy(maze)
   
    actual_cost_goals(objectives, maze_copy, dist_goals)

    #(fn, cur, nearest, furthest)
    q = PriorityQueue()
    cur_pos = maze.getStart()

    tup_obj = tuple(objectives)
    #to make path
    parents = {(cur_pos, tup_obj):((-1,-1), tup_obj)}
    
    #heuristic gn + hn
    #where gn is start node to cur_pos
    #hn is mst + min(manhattan distance)
    #fn, gn, pos, goals
    q.put((heuristic(cur_pos, tup_obj, dist_goals), 0 , cur_pos, tup_obj))
    
    path = []
    while not q.empty():
        cur_node = q.get()

        if len(cur_node[3]) == 0:
            #do the path thing here
            temp = (cur_node[2], cur_node[3])
            while temp != ((-1,-1), tup_obj):
                path.append(temp[0])
                temp = parents[temp]
            
            path.reverse()
            return path

        neighbors = maze.getNeighbors(cur_node[2][0], cur_node[2][1])
        
        for i in neighbors:
            n_goals = neighbors_goals(cur_node[3], i)

            
            if (i, n_goals) in parents:
                continue
            
            fn = heuristic(i, n_goals, dist_goals) + cur_node[1]+1


            q.put((fn, cur_node[1]+1 , i, n_goals))
            parents.update({(i, n_goals):(cur_node[2], cur_node[3])})

    
    return path


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    objectives = maze.getObjectives()
    copy_objectives = list(objectives)
    obj_len = len(objectives)

    cur_pos = maze.getStart()
    copy_cur_pos = cur_pos

    answer_path = []

    for i in range(obj_len):
        #set current position as start
        maze.setStart(cur_pos)
        #update objectives
        maze.setObjectives(objectives)

        #try eating closest food
        path = bfs(maze) 
        nearest_goal = path[-1]
        objectives.remove(nearest_goal)

        cur_pos = nearest_goal

        answer_path.extend(path)

    maze.setStart(copy_cur_pos)
    maze.setObjectives(copy_objectives)

    return answer_path
