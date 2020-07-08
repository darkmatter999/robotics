import sys
import os
import numpy as np

#these three functions implement a collision check of a given edge with a given obstacle

#the collision check is based on Jeffrey Thompson's effort (http://www.jeffreythompson.org/collision-detection/line-circle.php)

#as parameters of the main function **collision_check** the coordinates of either end of the 'line' (edge), the coordinates of the obstacle
#center and the diameter of the circular obstacle are used
def collision_check(x1, y1, x2, y2, cx, cy, d):
    #define the radius of the circular obstacle
    r = d/2
    #check if either the start or the end of the edge lies within the obstacle, if so, the collision is obvious, so the collision check can
    #be stopped
    inside_start_edge = node_in_obstacle(x1,y1,cx,cy,r)
    inside_end_edge = node_in_obstacle(x2,y2,cx,cy,r)
    if inside_start_edge or inside_end_edge:
        return True
    #if above check doesn't return True, the procedure is being resumed by defining the euclidean length of the edge
    distX = x1 - x2
    distY = y1 - y2
    length = np.sqrt((distX*distX) + (distY*distY))
    #define a point on the edge between node x and node y which lies closest to the center of the obstacle
    dot = (((cx-x1)*(x2-x1)) + ((cy-y1)*(y2-y1))) / length**2
    #define the coordinates of this point
    closestX = x1 + (dot * (x2-x1))
    closestY = y1 + (dot * (y2-y1))
    #check if the point is actually on the line segment. If not, there is no collision, since the edge is not affected by the obstacle
    point_on_line_segment = on_line_segment(x1,y1,x2,y2,closestX,closestY)
    if point_on_line_segment == False:
        return False
    #if above check doesn't return False, the collision check procedure is being resumed by defining the euclidean distance between
    #the closest point and the actual obstacle center
    distX = closestX - cx
    distY = closestY - cy
    distance = np.sqrt((distX*distX) + (distY*distY))
    #define a safety margin that prevents the robot from traversing the area too close to an obstacle
    #the radius simply becomes a little larger due to the addition of this margin
    safety_margin = r*0.2
    #if above deined distance is smaller (collision) or equal to (tangency) the obstacle radius + the safety margin, there is a collision
    if distance <= r + safety_margin:
        return True
    return False
#this function checks if a given node lies inside an obstacle, which leads to an obvious collision
def node_in_obstacle(px, py, cx, cy, d):
    r = d/2
    distX = px - cx
    distY = py - cy
    distance = np.sqrt((distX*distX) + (distY*distY))
    safety_margin = r*0.2
    if distance <= r + safety_margin:
        return True
    return False
#this function checks if the defined point closest to the obstacle center actually lies on the edge between node x and node y
def on_line_segment(x1, y1, x2, y2, closestX, closestY):
    d1 = np.sqrt((closestX-x1)**2 + (closestY-y1)**2)
    d2 = np.sqrt((closestX-x2)**2 + (closestY-y2)**2)
    lineLen = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #a small buffer range is defined to prevent the check from being too accurate due to the float numbers
    buffer = 0.3
    if d1+d2 >= lineLen-buffer and d1+d2 <=lineLen+buffer:
        return True
    return False

#this function defines the probabilistic roadmap as future parameter for A* search
def prm(obstacles):
    #load the obstacles parameter file into the script
    obstacles = np.loadtxt(obstacles, delimiter=',')
    #define the coordinates of the start node
    start_x = -0.5
    start_y = -0.5
    #define the coordinates of the goal node
    goal_x = 0.5
    goal_y = 0.5
    #define the number of 'nearest neighbors' for each node that are relevant for the finished 'edges' list and subsequent A* search
    k_nn = 70
    #hardcode the start node with its euclidean distance to the goal node
    nodes = np.array([[1,start_x,start_y,np.sqrt((goal_x-start_x)**2 + (goal_y-start_y)**2)]])
    #for the probabilistic roadmap, sample a range of nodes from a uniform distribution and save them into arrays, one for the random 
    #x value, the other for the y value. Indicate the length of the arrays, i.e. how many random nodes become created
    #the more random nodes, the higher the probability of actually finding a path
    sample_x = np.random.uniform(start_x, goal_y, 98)
    sample_y = np.random.uniform(start_x, goal_y, 98)
    #iteratively concatenate the samples to the 'nodes' list, hardcode the goal node, and save nodes to 'nodes.csv'
    for i in range(len(sample_x)):
        nodes = np.concatenate((nodes, np.array([[i+2, sample_x[i], sample_y[i], np.sqrt((goal_x-sample_x[i])**2 + (goal_y-sample_y[i])**2)]])), axis=0)
    nodes = np.concatenate((nodes, np.array([[len(sample_x)+2, goal_x, goal_y, np.sqrt((goal_x-goal_x)**2 + (goal_y-goal_y)**2)]])))
    np.savetxt("results/nodes.csv", nodes, fmt=['%d','%1.6f','%1.6f','%1.6f'], delimiter = ",")
    #initialize the full (unsorted, unfiltered) 'edges' array
    edges_full = np.array([[0, 0, 0]])
    i=0
    #loop through all nodes to connect each node with another and calculate the respective cost of traversal (as euclidean distance cost)
    #concatenate each edge with its two respective nodes and the cost to the 'edges_full' array
    while i < len(nodes):
        collision_sum = 0
        for k in range(len(nodes)):
            edges_full = np.concatenate((edges_full, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
            #for the current two nodes in the loop, iterate through all obstacles, carry out the collision check and penalize a given edge
            #if a collision has been found (set 1000 to be the cost and 'replace' the original edge cost with that penalty cost)
            for o in range(len(obstacles)):
                if nodes[i][0] != nodes[k][0]:
                    if collision_check(nodes[i][1], nodes[i][2], nodes[k][1], nodes[k][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                        collision_sum = collision_sum + 1000
                        edges_full[-1][2] = collision_sum
        i=i+1
    #delete the first edge, from the start node to the start node
    edges_full = np.delete(edges_full, 0, 0)
    #reshape the first edges array into an array consisting of <number of nodes> subarrays, each consisting of all edges for a given node
    edges_full = np.reshape(edges_full, (len(nodes), len(nodes), 3))
    #initialize the sorted 'edges' array
    edges_sorted = np.array([[0,0,0]])
    #loop through all subarrays of edges_full, sort them according to the edge cost in ascending order and keep only the nearest neighbors
    for e in range(len(edges_full)):
        edges_full[e] = edges_full[e][edges_full[e][:,2].argsort()]
        edges_sorted = np.concatenate((edges_sorted, np.array(edges_full[e][0:k_nn + 1])))
    #initialize the final (filtered) 'edges' array
    edges_filtered = np.array([[0,0,0]])
    #since the edges are bidirectional (e.g. the edge from 5 to 8 is the same as from 8 to 5), eliminate doubles and take out all penalized 
    #edges due to collisions
    #dissolve the 3D shape and represent all valid edges in one single array which then gets saved to a .csv file
    for d in range(len(edges_sorted)):
        if edges_sorted[d][0] > edges_sorted[d][1] and edges_sorted[d][2] < 1000:
            edges_filtered = np.concatenate((edges_filtered, np.array([edges_sorted[d]])))
    edges_filtered = np.delete(edges_filtered, 0, 0)
    #save edges_filtered to edges.csv, for use in A* search
    np.savetxt("results/edges.csv", edges_filtered, fmt=['%d','%d','%1.6f'], delimiter = ",")
    #invoke A* search with the just created 'nodes' and 'edges' .csv files and attempt to find a path
    os.system("python3.7 astar.py results/nodes.csv results/edges.csv")

if __name__ == "__main__":
    obstacles = str(sys.argv[1])
    prm(obstacles)
