import numpy as np

def astar(nodes, edges):

    nodes = np.loadtxt(nodes, delimiter=',')
    edges = np.loadtxt(edges, delimiter=',')


#print (f_array[:,1])
#print (f_array)

#print (split(f_array, f_array[:,1]==current)[0])
#print ([f_array[f_array[:,1]==current]])

    open_list = np.array([nodes[0][0]])
    past_cost = np.array([0])
    #path = np.array([])
    parent = {}
    etc = {}
    closed = np.array([])
    goal = max(nodes[:,0])
    for node in range(len(nodes[:,0])-1):
        past_cost=np.append(past_cost, np.inf)
    for node in nodes[:,0]:
        parent.update({int(node): np.inf})
    #print (parent)
    while (len(open_list)) > 0:
        current = open_list[0]
        #print (current)
        open_list = np.delete(open_list, 0)
        closed = np.append(closed, current)
        
        #path = np.append(path, current)
        #print (path)
        if current == goal:
            path = [goal, parent[len(parent)]]
            #print (path)
            i = len(parent)-1
            #print (list(parent.keys()))
            while i > 0:
                #print (parent[i])
                #print (path[-1])
                if list(parent.keys())[i] == path[-1]:
                    path.append(parent[path[-1]])
                i=i-1
            #return list(reversed(path))
            return parent

        else:
            neighbors = edges[edges[:,1]==current]
            #print (neighbors)
            for neighbor in range(len(neighbors)):
                if np.int(neighbors[neighbor][0]) not in closed:
                    #print (neighbor)
                    tentative_past_cost = past_cost[np.int(current-1)] + neighbors[neighbor][2]
                    #print (tentative_past_cost)
                    #print(past_cost[int(neighbor[0][0])])
                    if tentative_past_cost < past_cost[int(neighbors[neighbor][0]-1)]:
                        past_cost[np.int(neighbors[neighbor][0]-1)] = tentative_past_cost
                        parent[np.int(neighbors[neighbor][0])] = current
                        #print(past_cost[int(neighbor[0][0])])
                        etc.update({int(neighbors[neighbor][0]): past_cost[int(neighbors[neighbor][0]-1)] + nodes[:,3][int(neighbors[neighbor][0]-1)]})
                        #print (etc)
                        #estimated_total_cost = past_cost[int(neighbors[neighbor][0]-1)] + nodes[:,3][int(neighbors[neighbor][0]-1)]
                        #print (estimated_total_cost)
                        #open_list = np.insert(open_list, np.searchsorted(open_list, estimated_total_cost), int(neighbors[neighbor][0]))
                        #print (open_list)
                        sorted_etc = sorted(etc.items(), key=lambda x: x[1])
                        #print (sorted_etc)
                        for i in sorted_etc:
                            open_list = np.append(open_list, i[0])
                            #print (open_list)
                        
    return ('failure')

print (astar("nodes.csv", "edges.csv"))

#check float - int issue
#retry np.searchsorted
    