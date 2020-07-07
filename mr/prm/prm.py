import numpy as np

obstacles = np.loadtxt("obstacles - first.csv", delimiter=',')
#print (obstacles)

start_distr = -0.5
end_distr = 0.5

start_x = -0.5
start_y = -0.5

goal_x = 0.5
goal_y = 0.5

k_nn = 3

nodes = np.array([[1,-0.5,-0.5,np.sqrt((goal_x-start_x)**2 + (goal_y-start_y)**2)]])

sample_x = np.random.uniform(start_distr, end_distr, 30)
sample_y = np.random.uniform(start_distr, end_distr, 30)

for i in range(len(sample_x)):
    nodes = np.concatenate((nodes, np.array([[i+2, sample_x[i], sample_y[i], np.sqrt((goal_x-sample_x[i])**2 + (goal_y-sample_y[i])**2)]])), axis=0)

nodes = np.concatenate((nodes, np.array([[len(sample_x)+2, goal_x, goal_y, np.sqrt((goal_x-goal_x)**2 + (goal_y-goal_y)**2)]])))
nodes = np.array([[1,-0.5,-0.5,1.4142],[2,-0.09,-0.4,1.0762],[3,-0.285,-0.305,1.1244],[4,0.0575,-0.225,0.8494],[5,-0.0525,-0.0175,0.7604],[6,-0.37,0.3,0.8927],[7,0.3525,-0.0525,0.5719],[8,0.0625,0.255,0.5014],[9,-0.1,0.3725,0.6134],[10,0.4275,0.195,0.3135],[11,0.345,0.3525,0.214],[12,0.5,0.5,0]])
#nodes = np.loadtxt("nodes_test.csv", delimiter=',')
#print (nodes)
np.savetxt("results/nodes.csv", nodes, fmt=['%d','%1.6f','%1.6f','%1.6f'], delimiter = ",")


'''
def collision_check(ax,bx,ay,by,cx,cy,r):
    #compute euclidean distance of the two points
    eucl_dist_p = np.sqrt((bx-ax)**2 + (by-ay)**2)
    #compute direction vector
    dx = (bx-ax) / eucl_dist_p
    dy = (by-ay) / eucl_dist_p
    #the point of line ab closest to the obstacle (circle) center needs to be found 
    closest_to_center = dx*(cx-ax) + dy*(cy-ay)
    #compute the coordinates of the point closest_to_center
    closest_x = closest_to_center * dx + ax
    closest_y = closest_to_center * dy + ay
    #compute the euclidean distance between closest_to_center and the circle center
    eucl_dist_c = np.sqrt((closest_x-cx)**2 + (closest_y-cy)**2)
    if eucl_dist_c < r:
        dt = np.sqrt(r**2 - eucl_dist_c**2)
        t1 = closest_to_center - dt
        t2 = closest_to_center + dt
        if t1 < 0 or t1 > 1 and t2 < 0 or t2 > 1:
            return False
        else:
            return True
    else:
        return False

'''

#collision check new

def collision_check(x2, x1, y2, y1, cx, cy, r):
    inside1 = pointCircle(x1,y1, cx,cy,r)
    inside2 = pointCircle(x2,y2, cx,cy,r)
    if inside1 or inside2:
        return True
    
    distX = x1 - x2
    distY = y1 - y2
    length = np.sqrt((distX*distX) + (distY*distY))

    dot = (((cx-x1)*(x2-x1)) + ((cy-y1)*(y2-y1))) / length**2

    closestX = x1 + (dot * (x2-x1))
    closestY = y1 + (dot * (y2-y1))

    onSegment = linePoint(x1,y1,x2,y2, closestX,closestY)
    if onSegment == False:
        #print (onSegment)
        return False
    
    distX = closestX - cx
    distY = closestY - cy
    distance = np.sqrt((distX*distX) + (distY*distY))

    if distance <= r:
        return True
    return False

def pointCircle(px, py, cx, cy, r):
    distX = px - cx
    distY = py - cy
    distance = np.sqrt((distX*distX) + (distY*distY))

    if distance <= r:
        return True
    return False

def linePoint(x2, x1, y2, y1, px, py):
    d1 = np.sqrt((px-py)**2 + (x1-y1)**2)
    #print (d1)
    d2 = np.sqrt((px-py)**2 + (x2-y2)**2)
    #print (d2)
    lineLen = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #print (lineLen)

    buffer = 0.01

    if d1+d2 >= lineLen-buffer and d1+d2 <=lineLen+buffer:
        return True
    return False



print (collision_check(-0.052500,-0.5,-0.017500,-0.5,-0.285,-0.075,0.33))
#print (collision_check(-0.052500,-0.5,-0.017500,-0.5,0.365,-0.295,0.27))
#print (collision_check(-0.052500,-0.5,-0.017500,-0.5,0.205,0.155,0.15)) 


cx = 0
cy = 0
r = 0.3

edges = np.array([[0, 0, 0]])
i=0

'''
while i < len(nodes):
    for k in range(len(nodes)):
        for o in range(len(obstacles)):
            if collision_check(nodes[k][1], nodes[i][1], nodes[k][2], nodes[i][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                edges = np.concatenate((edges, np.array([[nodes[k][0], nodes[i][0], 1000]])))
            else:
                edges = np.concatenate((edges, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
    i=i+1

'''

while i < len(nodes):
    for k in range(len(nodes)):
        edges = np.concatenate((edges, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
        #print (edges[1])
        
        for o in range(len(obstacles)):
            #print (collision_check(nodes[k][1], nodes[i][1], nodes[k][2], nodes[i][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]))
            if collision_check(nodes[k][1], nodes[i][1], nodes[k][2], nodes[i][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                #print (edges[-1])
                edges[-1][2] = edges[-1][2] + 1000
                #print (edges[-1])
        
    i=i+1
'''
edges_new = np.array([[0, 0, 0]])

while i < len(nodes):
    
    for k in range(len(nodes)):
        collision_sum = 0
        edges = np.concatenate((edges, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
        #print (edges[1])
        
        
        for o in range(len(obstacles)):
            
            if collision_check(nodes[k][1], nodes[i][1], nodes[k][2], nodes[i][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                collision_sum = collision_sum + 1000
                #print (edges_new)
            if collision_sum < 1000:
                edges_new = np.concatenate((edges_new, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
        
    i=i+1
'''

#print (edges)

edges = np.delete(edges, 0, 0)
edges = np.reshape(edges, (len(nodes), len(nodes), 3))

#print (edges)

edges1 = np.array([[0,0,0]])


#for e in range(len(edges)):
    #edges[e] = edges[e][edges[e][:,2].argsort()]
    #edges1 = np.concatenate((edges1, np.array(edges[e][0:k_nn + 1])))
    #edges1 = np.concatenate((edges1, np.array(edges[e])))
    #edges1 = np.concatenate((edges1, np.array(edges[e][1:len(edges[e])])))


#print (nodes[int(edges[5][2][1])-1][1])
#print (edges[5][2][1])

'''
for e in range(len(edges)):
    for o in range(len(obstacles)):
        for p in range(len(edges[e])):
            if collision_check(nodes1[int(edges[e][p][1])-1][1], nodes1[int(edges[e][p][0])-1][1], nodes1[int(edges[e][p][1])-1][2], nodes1[int(edges[e][p][0])-1][2], edges[e][p][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                edges[e][p][2] = 1000
    edges[e] = edges[e][edges[e][:,2].argsort()]
    print (edges)
    edges1 = np.concatenate((edges1, np.array(edges[e][1:k_nn + 1])))
    #print (edges1)
'''


#eliminate doubles

#print (edges1)
#edges2 = np.array([[0,0,0]])
edges3 = np.array([[0,0,0]])

#for d in range(len(edges1)):
    #if edges1[d][0] > edges1[d][1] and edges1[d][2] < 1000:
    #if edges1[d][2] < 1000:
        #edges2 = np.concatenate((edges2, np.array([edges1[d]])))

for d in range(len(edges)):
    if edges1[d][0] > edges1[d][1] and edges1[d][2] < 1000:
    #if edges[d][0] != edges[d][1] and edges[d][0] > edges[d][1]:
        edges1 = np.concatenate((edges1, np.array([edges1[d]])))

edges3 = np.delete(edges3, 0, 0)
#print (edges2[1][1])
#print (nodes1[int(edges2[17][1])][1])
#print (edges2[1][2])

#print (collision_check(0.5, 0.3, 0.4, -0.2, 0.42585, 0.5, 0.3, 0.2))
#print (len(edges2))

#print (edges2)
#bx ax by ay
#2,0.023011,0.325946,0.507753
#12,0.500000,0.500000,0.000000

'''
print (collision_check(0.063701,0.5,0.325946,0.5,0.0, 0.0, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,0.0, 0.1, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,0.3, 0.2, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,-0.3, -0.2, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,-0.1, -0.4, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,-0.2, 0.3, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,0.3, -0.3, 0.2))
print (collision_check(0.023011,0.5,0.325946,0.5,0.1, 0.4, 0.2))

print (collision_check(0.063701,-0.5,-0.438482,-0.5,0.0, 0.0, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,0.0, 0.1, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,0.3, 0.2, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,-0.3, -0.2, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,-0.1, -0.4, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,-0.2, 0.3, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,0.3, -0.3, 0.2))
print (collision_check(0.063701,-0.5,-0.438482,-0.5,0.1, 0.4, 0.2))


j=0
while j < len(edges2):
    #print (nodes[int(edges2[j][0])-1][1], nodes[int(edges2[j][1])-1][1], nodes[int(edges2[j][0])-1][2], nodes[int(edges2[j][1])-1][2], edges2[j][2], obstacles[o][0], obstacles[o][1], obstacles[o][2])
    for o in range(len(obstacles)):
        if collision_check(nodes[int(edges2[j][1])][1], nodes[int(edges2[j][0])][1], nodes[int(edges2[j][1])][2], nodes[int(edges2[j][0])][2], edges[j][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
            #print (nodes[int(edges2[j][0])-1][1], nodes[int(edges2[j][1])-1][1], nodes[int(edges2[j][0])-1][2], nodes[int(edges2[j][1])-1][2], edges2[j][2], obstacles[o][0], obstacles[o][1], obstacles[o][2])
            edges2[j][2] = 1000
    j=j+1
'''

np.savetxt("results/edges.csv", edges1, fmt=['%d','%d','%1.6f'], delimiter = ",")
#print (edges2)

#nodes1[edges[e][1]]