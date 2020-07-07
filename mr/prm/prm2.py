import numpy as np

#these functions need to be redesigned
def collision_check(x1, y1, x2, y2, cx, cy, r):
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

    onSegment = linePoint(x1,y1,x2,y2,closestX,closestY)
    if onSegment == False:
        return False
    
    distX = closestX - cx
    distY = closestY - cy
    distance = np.sqrt((distX*distX) + (distY*distY))

    if distance <= r/2:
        return True
    return False

def pointCircle(px, py, cx, cy, r):
    distX = px - cx
    distY = py - cy
    distance = np.sqrt((distX*distX) + (distY*distY))
    
    if distance <= r/2:
        return True
    return False

def linePoint(x1, y1, x2, y2, px, py):
    d1 = np.sqrt((px-x1)**2 + (py-y1)**2)
    d2 = np.sqrt((px-x2)**2 + (py-y2)**2)
    lineLen = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    buffer = 0.4

    if d1+d2 >= lineLen-buffer and d1+d2 <=lineLen+buffer:
        return True
    return False

#collision check tests

#print (collision_check(-0.5,-0.5,0.5,0.5,-0.285,-0.075,0.33))
#print (collision_check(-0.5,-0.5,0.5,0.5,0.365,-0.295,0.27))
#print (collision_check(-0.5,-0.5,0.5,0.5,0.205,0.155,0.15)) 

#print (collision_check(-0.090000,-0.400000,-0.052500,-0.017500,-0.285,-0.075,0.33))
#print (collision_check(-0.090000,-0.400000,-0.052500,-0.017500,0.365,-0.295,0.27))
#print (collision_check(-0.090000,-0.400000,-0.052500,-0.017500,0.205,0.155,0.15)) 

obstacles = np.loadtxt("results/obstacles.csv", delimiter=',')

start_distr = -0.5
end_distr = 0.5

start_x = -0.5
start_y = -0.5

goal_x = 0.5
goal_y = 0.5

k_nn = 7

nodes = np.array([[1,-0.5,-0.5,np.sqrt((goal_x-start_x)**2 + (goal_y-start_y)**2)]])

sample_x = np.random.uniform(start_distr, end_distr, 28)
sample_y = np.random.uniform(start_distr, end_distr, 28)

for i in range(len(sample_x)):
    nodes = np.concatenate((nodes, np.array([[i+2, sample_x[i], sample_y[i], np.sqrt((goal_x-sample_x[i])**2 + (goal_y-sample_y[i])**2)]])), axis=0)

nodes = np.concatenate((nodes, np.array([[len(sample_x)+2, goal_x, goal_y, np.sqrt((goal_x-goal_x)**2 + (goal_y-goal_y)**2)]])))
#nodes = np.array([[1,-0.5,-0.5,1.4142],[2,-0.09,-0.4,1.0762],[3,-0.285,-0.305,1.1244],[4,0.0575,-0.225,0.8494],[5,-0.0525,-0.0175,0.7604],[6,-0.37,0.3,0.8927],[7,0.3525,-0.0525,0.5719],[8,0.0625,0.255,0.5014],[9,-0.1,0.3725,0.6134],[10,0.4275,0.195,0.3135],[11,0.345,0.3525,0.214],[12,0.5,0.5,0]])

np.savetxt("results/nodes.csv", nodes, fmt=['%d','%1.6f','%1.6f','%1.6f'], delimiter = ",")

edges = np.array([[0, 0, 0]])
i=0

while i < len(nodes):
    collision_sum = 0
    for k in range(len(nodes)):
        edges = np.concatenate((edges, np.array([[nodes[k][0], nodes[i][0], np.sqrt((nodes[k][1]-nodes[i][1])**2 + (nodes[k][2]-nodes[i][2])**2)]])))
        
        for o in range(len(obstacles)):
            if nodes[i][0] != nodes[k][0]:
                if collision_check(nodes[i][1], nodes[i][2], nodes[k][1], nodes[k][2], obstacles[o][0], obstacles[o][1], obstacles[o][2]):
                    collision_sum = collision_sum + 1000
                    edges[-1][2] = collision_sum
        
    i=i+1

edges = np.delete(edges, 0, 0)
edges = np.reshape(edges, (len(nodes), len(nodes), 3))

edges1 = np.array([[0,0,0]])

#is it necessary to have another array for the operation of sorting and 'shorting'?
for e in range(len(edges)):
    edges[e] = edges[e][edges[e][:,2].argsort()]
    edges1 = np.concatenate((edges1, np.array(edges[e][0:k_nn + 1])))
    
edges2 = np.array([[0,0,0]])

for d in range(len(edges1)):
    if edges1[d][0] > edges1[d][1] and edges1[d][2] < 1000:
        edges2 = np.concatenate((edges2, np.array([edges1[d]])))

edges2 = np.delete(edges2, 0, 0)

np.savetxt("results/edges.csv", edges2, fmt=['%d','%d','%1.6f'], delimiter = ",")
