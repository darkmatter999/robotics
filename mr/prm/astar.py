'''
***A* ALGORITHM IMPLEMENTED IN PYTHON***

This function implements the A* search algorithm in Python. It finds the cheapest path from a given start node to a given end node in
a weighted graph. The given parameters are an array of nodes, with the numbered node and the heuristic cost (e.g. direct line from that
node to the goal), and an array of edges, indicating the direct ('neighborly') paths from one given node to another and their 
respective path costs.

The algorithm returns the cheapest path found from the start node to the goal node. It is assumed that the path starts at node 1 and ends
at the highest-numbered node.

'''

import numpy as np

def astar(nodes, edges):

    #load the .csv input parameter files 'nodes' and 'edges' and convert to Numpy arrays
    nodes = np.loadtxt(nodes, delimiter=',')
    edges = np.loadtxt(edges, delimiter=',')
    
    #initialize the list of nodes to be explored with the start node
    open_list = np.array([nodes[0][0]])
    #initialize the cost so far accrued for a given node with the cost from the start node to the start node, which is 0
    past_cost = np.array([0])
    #initialize an empty dictionary for each node a link to the node preceding it in the shortest path found so far from the start node to that node.
    parent = {}
    #initialize an empty dictionary in which the estimated total cost of each 'open' nodes is stored and later sorted in ascending order
    estimated_total_cost = {}
    #initialize an array for the nodes not to be explored anymore
    closed = np.array([])
    #set the goal node to be the last (highest-numbered) node in the 'nodes' array
    goal = int(max(nodes[:,0]))
    #in the 'past_cost' array, set the costs for node 2 to node N to infinity, since the real cost is not known yet
    for node in range(len(nodes[:,0])-1):
        past_cost=np.append(past_cost, np.inf)
    #in the 'parent' dictionary, set the parent of each node to infinity, since the real parent is not known yet
    for node in nodes[:,0]:
        parent.update({int(node): np.inf})
    
    #loop through the nodes to find the minimum-cost path until the goal is found or there are no more nodes to further explore
    while (len(open_list)) > 0:
        #set a variable 'current' to be the first node in 'open_list'
        current = open_list[0]
        #remove the first node from 'open_list'
        open_list = np.delete(open_list, 0)
        #append 'current' to the 'closed' list
        closed = np.append(closed, current)
        
        #if 'current' is the goal node, reconstruct the path from the 'parent' dictionary of nodes and their optimal parents and return it
        if current == goal:
            path = np.array([goal])
            i = len(parent)-1
            while i > 0:
                if list(parent.keys())[i] == path[-1]:
                    path = np.append(path, int(parent[path[-1]]))
                    #print (path)
                i=i-1
            path = np.flip(path)
            path = np.reshape(path, (1,len(path)))
            path = np.savetxt('results/path.csv', path, fmt="%d", delimiter=',')
            return (print('Success! Path saved to "path.csv"'), path)
           
        else:
            #as long as the goal is not found, explore the neighbors of 'current'

            #define the neighbors of current and store them in an array
            neighbors = edges[edges[:,1]==current]
            #loop through all individual neighbors of the 'current' node
            for neighbor in range(len(neighbors)):
                if int(neighbors[neighbor][0]) not in closed:
                    #define 'tentative_past_cost' as the past cost of the 'current' node + the cost of traveling from the 'current' node to the neighbor
                    tentative_past_cost = past_cost[int(current-1)] + neighbors[neighbor][2]
                    #set 'tentative_past_cost' to be the past cost of the neighbor if it is lower than the neighbor's past cost
                    if tentative_past_cost < past_cost[int(neighbors[neighbor][0]-1)]:
                        past_cost[int(neighbors[neighbor][0]-1)] = tentative_past_cost
                        #set the parent of the neighbor to be the 'current' node
                        parent[int(neighbors[neighbor][0])] = current
                        #in 'estimated_total_cost', store the neighbor's past cost summed with the heuristic cost of the neighbor to the goal
                        estimated_total_cost.update({int(neighbors[neighbor][0]): past_cost[int(neighbors[neighbor][0]-1)] + nodes[:,3][int(neighbors[neighbor][0]-1)]})
                        #sort 'estimated_total_cost' in ascending order
                        sorted_estimated_total_cost = sorted(estimated_total_cost.items(), key=lambda x: x[1])
                        #append the entries contained in 'estimated_total_cost' to 'open_list' and keep iterating
                        for i in sorted_estimated_total_cost:
                            open_list = np.append(open_list, i[0])

    #return 'failure' if there is no available path from the start node to the goal node
    return (print('Failure: No available path'))

#call the function with the two required input parameters 'nodes' and 'edges'
astar("results/nodes.csv", "results/edges.csv")


    