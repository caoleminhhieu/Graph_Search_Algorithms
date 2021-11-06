import pygame
from constant_variable import display_height, display_width, fps_speed
import graphUI
from utils import draw_graph
from node_color import white, yellow, black, red, blue, purple, orange, green
import math
import numpy as np
from copy import deepcopy
"""
Feel free print graph, edges to console to get more understand input.
Do not change input parameters
Create new function/file if necessary
"""

def Solution(graph, edges, edge_id, goal, parent, weightedMatrix):
    child = goal
    graph[child][2] = white
    graph[child][3] = purple
    graphUI.updateUI()

    PathCost = 0
    while parent[child] != -1:
        PathCost = PathCost + weightedMatrix[parent[child]][child]
        edges[edge_id(parent[child],child)][1] = green
        child = parent[child]
        graphUI.updateUI()
 
    if child != goal:
        graph[child][3] = orange
    graphUI.updateUI()
    return PathCost

def BFS(graph, edges, edge_id, start, goal):
    """
    BFS search
    """
    # TODO: your code
    # Start and Goal are at the same place.
    print("Implement BFS algorithm.")
    parent = {}
    parent[start] = -1
    n = len(graph)
    weightedMatrix = np.ones((n,n))

    if start == goal:
        return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))

    #-----#-----#-----#-----#-----#-----
    frontier = []
    frontier.append(start)
    explored = set()

    while True:
        if len(frontier) == 0:
            print("Can't find the goal!!!")
            return -1

        node = frontier.pop(0)
        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored.add(node)

        for successor in graph[node][1]:
            if (successor not in explored) and (successor not in frontier):
                parent[successor] = node
                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()

                if successor == goal:
                    return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))
                else:
                    frontier.append(successor)
        graph[node][3] = blue
            
    print("Implement BFS algorithm.")
    pass


def DFS(graph, edges, edge_id, start, goal):
    """
    DFS search
    """
    # TODO: 
    print("Implement DFS algorithm.")
    n = len(graph)
    weightedMatrix = np.ones((n,n))
    parent = {}
    parent[start] = -1

    if start == goal:
       return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))


    #-----#-----#-----#-----#-----#-----
    frontier = []
    frontier.append(start)
    explored = set()

    while True:
        if len(frontier) == 0:
            print("Can't find the goal!!!")
            return -1

        node = frontier.pop()
        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored.add(node)

        for successor in graph[node][1]:
            if (successor not in explored) and (successor not in frontier):
                parent[successor] = node
                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()

                if successor == goal:
                    return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))
                else:
                    frontier.append(successor)
        graph[node][3] = blue
            
    print("Implement DFS algorithm.")
    pass

def ComputeweightedMatrix(graph):
    n = len(graph)
    weightedMatrix = np.zeros((n,n))
    for start in range(n):
        for end in range(n):
            if end in graph[start][1]:
                weightedMatrix[start][end] = math.sqrt((graph[start][0][0] - graph[end][0][0])**2 + (graph[start][0][1] - graph[end][0][1])**2)
            else:
                weightedMatrix[start][end] = math.inf
        
    return weightedMatrix

def UCS(graph, edges, edge_id, start, goal):
    """
    Uniform Cost Search search
    """
    # TODO: your code
    print("Implement Uniform Cost Search algorithm.")
    parent = {}
    parent[start] = -1

    #-----#-----#-----#-----#-----#-----
    weightedMatrix = ComputeweightedMatrix(graph)
    frontier = {}
    frontier[start] = 0 # 0: Euclidean Distance
    explored = set()

    while True:
        if len(frontier) == 0:
            print("Can't find the goal!!!")
            return -1

        node = min(frontier, key = frontier.get)
        node_distance = frontier.pop(node)

        if node == goal:
            return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))

        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored.add(node)

        index = 0
        for successor in graph[node][1]:
            distance = node_distance + weightedMatrix[node][successor]
            if (successor not in explored) and (successor not in frontier):
                parent[successor] = node
                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()

                parent[successor] = node
                frontier[successor] = distance

            elif successor in frontier and distance < frontier[successor]:
                frontier[successor] = distance
                parent[successor] = node
            index += 1
        graph[node][3] = blue
    print("Implement Uniform Cost Search algorithm.")
    pass

def heuristicsFunc(graph, goal):
    heuF = []
    n = len(graph)
    for node in range(n):
        EuclideanDistance = math.sqrt((graph[node][0][0] - graph[goal][0][0])**2 + (graph[node][0][1] - graph[goal][0][1])**2)
        heuF.append(EuclideanDistance)
    return heuF
def AStar(graph, edges, edge_id, start, goal):
    """
    A star search
    """
    # TODO: your code
    print("Implement A* algorithm.")
    parent = {}
    parent[start] = -1

    #-----#-----#-----#-----#-----#-----
    weightedMatrix = ComputeweightedMatrix(graph)
    heuF = heuristicsFunc(graph, goal)

    frontier = {}
    frontier[start] = heuF[start] # 0: Euclidean Distance
    explored = set()

    while True:
        if len(frontier) == 0:
            print("Can't find the goal!!!")
            return -1

        node = min(frontier, key = frontier.get)
        node_distance = frontier.pop(node)

        if node == goal:
           return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))

        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored.add(node)

        index = 0
        for successor in graph[node][1]:
            distance = node_distance + weightedMatrix[node][successor] - heuF[node] + heuF[successor]
            if (successor not in explored) and (successor not in frontier):
                parent[successor] = node
                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()

                parent[successor] = node
                frontier[successor] = distance

            elif successor in frontier and distance < frontier[successor]:
                frontier[successor] = distance
                parent[successor] = node
                edges[edge_id(node,successor)][1] = white
            index += 1
        graph[node][3] = blue
    print("Implement A* algorithm.")
    pass

def GreedySearch(graph, edges, edge_id, start, goal):
       # TODO: your code
    print("Implement Greedy Search algorithm.")
    parent = {}
    parent[start] = -1

    #-----#-----#-----#-----#-----#-----
    heuF = heuristicsFunc(graph, goal)
    weightedMatrix = ComputeweightedMatrix(graph)
    explored = set()
    node = start

    while True:

        if node == goal:
            return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent,weightedMatrix))

        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored.add(node)

        minHeuDistance = math.inf
        nodeChosen = -1
        for successor in graph[node][1]:
            if (successor not in explored) and (heuF[successor] < minHeuDistance):
                nodeChosen = successor
                minHeuDistance = heuF[nodeChosen]

        if nodeChosen != -1:
            parent[nodeChosen] = node
            edges[edge_id(node,nodeChosen)][1] = white
            graph[nodeChosen][2] = white
            graph[nodeChosen][3] = red
            graphUI.updateUI()
        else:
            print("Can't find the goal!!!")
            return -1
        graph[node][3] = blue
        node = nodeChosen
    pass

def BidirectionalSearch(graph, edges, edge_id, start, goal):
     # TODO: your code
    # Start and Goal are at the same place.
    print("Implement Bidirectional Search algorithm.")
    parent_forward = {}
    parent_forward[start] = -1

    parent_backward = {}
    parent_backward[goal] = -1
    n = len(graph)
    weightedMatrix = np.ones((n,n))

    if start == goal:
        return print("The length of the path:",Solution(graph, edges, edge_id,goal, parent_forward,weightedMatrix))

    #-----#-----#-----#-----#-----#-----
    frontier_forward = []
    frontier_forward.append(start)
    explored_forward = set()

    frontier_backward = []
    frontier_backward.append(goal)
    explored_backward = set()

    while True:
        if len(frontier_backward) == 0 and len(frontier_forward) == 0:
            print("Can't find the goal!!!")
            return -1

        #Foward Search
        node = frontier_forward.pop(0)
        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored_forward.add(node)

        for successor in graph[node][1]:
            if (successor not in explored_forward) and (successor not in frontier_forward):
                parent_forward[successor] = node
                if successor in frontier_backward:
                    return print("The length of the path:",Solution(graph, edges, edge_id,successor, parent_forward,weightedMatrix)
                                                        + Solution(graph, edges, edge_id,successor, parent_backward,weightedMatrix))

                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()
                frontier_forward.append(successor)
        graph[node][3] = blue

        #Backward Search
        node = frontier_backward.pop(0)
        graph[node][3] = yellow
        graph[node][2] = white
        graphUI.updateUI()

        explored_backward.add(node)

        for successor in graph[node][1]:
            if (successor not in explored_backward) and (successor not in frontier_backward):
                parent_backward[successor] = node
                if successor in frontier_forward:
                    return print("The length of the path:",Solution(graph, edges, edge_id,successor, parent_forward,weightedMatrix)
                                                        + Solution(graph, edges, edge_id,successor, parent_backward,weightedMatrix))

                edges[edge_id(node,successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                graphUI.updateUI()
                frontier_backward.append(successor)

        graph[node][3] = blue
    pass


def DFS_limitedDepth(graph, edges, edge_id, start, goal, limited):
    """
    DFS search
    """
    # TODO: 
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((display_width, display_height))
    font = pygame.font.Font(pygame.font.get_default_font(), 25)
    print("Implement Depth-Limited-Search algorithm: limited =", limited)
    n = len(graph)
    weightedMatrix = np.ones((n,n))
    parent = {}
    parent[start] = -1

    if start == goal:
        graph[goal][3] = purple
        graph[goal][2] = white
        draw_graph(screen, font, graph, edges)
        pygame.display.update()
        clock.tick(fps_speed)
        return True


    #-----#-----#-----#-----#-----#-----
    frontier = []
    frontier.append([start,0])
    explored = set()

    while True:
        if len(frontier) == 0 :
            print("Can't find the goal!!!")
            return False

        node = frontier.pop()
        depth = node[1]
        explored.add(node[0])
        if depth >= limited:
            continue
        graph[node[0]][3] = yellow
        graph[node[0]][2] = white
        #Draw copy graph
        draw_graph(screen, font, graph, edges)
        pygame.display.update()
        clock.tick(fps_speed)


        for successor in graph[node[0]][1]:
            if (successor not in explored) and (successor not in [i[0] for i in frontier]):
                parent[successor] = node[0]
                edges[edge_id(node[0],successor)][1] = white
                graph[successor][2] = white
                graph[successor][3] = red
                
                #Draw copy graph
                draw_graph(screen, font, graph, edges)
                pygame.display.update()
                clock.tick(fps_speed)

                if successor == goal:
                    child = goal
                    graph[child][2] = white
                    graph[child][3] = purple
                    draw_graph(screen, font, graph, edges)
                    pygame.display.update()
                    clock.tick(fps_speed)

                    PathCost = 0
                    while parent[child] != -1:
                        PathCost = PathCost + weightedMatrix[parent[child]][child]
                        edges[edge_id(parent[child],child)][1] = green
                        child = parent[child]
                        draw_graph(screen, font, graph, edges)
                        pygame.display.update()
                        clock.tick(fps_speed)
 
                    if child != goal:
                        graph[child][3] = orange
                        draw_graph(screen, font, graph, edges)
                        pygame.display.update()
                        clock.tick(fps_speed)
                    return True
                frontier.append([successor,depth + 1])
        graph[node[0]][3] = blue

    pass

#Iterative - Deepening - Search
def IDS (graph, edges, edge_id, start, goal): 

    n = len(graph)
    weightedMatrix = np.ones((n,n))
    for i in range(n):
        graph_cpy = deepcopy(graph)
        edges_cpy = deepcopy(edges)

        found = DFS_limitedDepth (graph_cpy, edges_cpy, edge_id, start, goal, i)

        if found == True:
            return i


        
    return -1

def BeamSearch(graph, edges, edge_id, start, goal):
    k = 2
    #k = int(input("Input k beam states: "))
    weightedMatrix = ComputeweightedMatrix(graph)
    parent={}
    parent[start] = -1

    if start == goal:
        return print("The length of the path:", Solution(graph, edges, edge_id, goal, parent, weightedMatrix))

    frontier = []
    frontier.append(start)
    explored = set()
    heuF = heuristicsFunc(graph, goal)
    print(heuF)

    while True:
        n = len(frontier)
        if n == 0:
            print("Can't find the goal!!!")
            return -1

        for i in range(n):
            node = frontier.pop(0)
            graph[node][3] = yellow
            graph[node][2] = white
            graphUI.updateUI()
            for successor in graph[node][1]:
                if (successor not in explored) and (successor not in frontier) :
                    parent[successor] = node
                    graph[successor][3] = red
                    graph[successor][2] = white
                    edges[edge_id(node,successor)][1] = white
                    graphUI.updateUI()

                    if successor == goal:
                        return print("The length of the path:", Solution(graph, edges, edge_id, goal, parent, weightedMatrix))

                    frontier.append(successor)

            explored.add(node)
            graph[node][3] = blue

        n = len(frontier)
        heuF_2 = []
        for i in range(n):
            heuF_2.append(heuF[frontier[i]])
        frontier = [x for _,x in sorted(zip(heuF_2,frontier))]
        while len(frontier) > k:
            node = frontier.pop()
            graph[node][3] = blue
            graphUI.updateUI()
            explored.add(node)



        

def example_func(graph, edges, edge_id, start, goal):
    """
    This function is successorust show some basic feature that you can use your prosuccessorect.
    @param graph: list - contain information of graph (same value as global_graph)
                    list of obsuccessorect:
                     [0] : (x,y) coordinate in UI
                     [1] : adsuccessoracent node indexes
                     [2] : node edge color
                     [3] : node fill color
                Ex: graph = [
                                [
                                    (139, 140),             # position of node when draw on UI
                                    [1, 2],                 # list of adsuccessoracent node
                                    (100, 100, 100),        # grey - node edged color
                                    (0, 0, 0)               # black - node fill color
                                ],
                                [(312, 224), [0, 4, 2, 3], (100, 100, 100), (0, 0, 0)],
                                ...
                            ]
                It means this graph has Node 0 links to Node 1 and Node 2.
                Node 1 links to Node 0,2,3 and 4.
    @param edges: dict - dictionary of edge_id: [(n1,n2), color]. Ex: edges[edge_id(0,1)] = [(0,1), (0,0,0)] : set color
                    of edge from Node 0 to Node 1 is black.
    @param edge_id: id of each edge between two nodes. Ex: edge_id(0, 1) : id edge of two Node 0 and Node 1
    @param start: int - start vertices/node
    @param goal: int - vertices/node to search
    @return:
    """

    # Ex1: Set all edge from Node 1 to Adsuccessoracency node of Node 1 is green edges.
    node_1 = graph[1]
    for adsuccessoracency_node in node_1[1]:
        edges[edge_id(1, adsuccessoracency_node)][1] = green
        graphUI.updateUI()

    # Ex2: Set color of Node 2 is Red
    graph[2][3] = red
    graphUI.updateUI()

    # Ex3: Set all edge between node in a array.
    path = [4, 7, 9]  # -> set edge from 4-7, 7-9 is blue
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = blue
        graphUI.updateUI()
