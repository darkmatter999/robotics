#This program evaluates the average rate of change of a function at an arbitrary interval.
#It takes a function of the form '[int1]x^2+[int2]x+[int3]' (e.g. 5x^2-4x+3) as input
#Further inputs are the start point of the interval, the stop point and two arbitrarily chosen real numbers as points
#Below is an example of how the algorithm works:

'''
FUNCTION: 4x^2 - 5
INTERVAL: (-4, -4+h) where h != 0

we need to find a function f(h), which is a line. It represents the average rate of change of above function at above interval.
This function f(h) is of the form y=mx+b, the general slope formula.

evaluation at x=-4: f(x) = 59
if h=1, then x = -4+h, i.e. -4+1 = -3 -->
evaluation at x=-3: f(x) = 31

now we evaluate the slope of the interval (-4, -3) -->

(dy / dx)
31-59
---
-3-(-4)

=

-28/1 = -28

Therefore, f(h) at h=1 is -28.

Now we need a second point of the function f(h)

We set h to 3 -->
evaluation at x=-4: f(x) = 59
if h=3, then x = -4+h, i.e. -4+3 = -1 -->
evaluation at x=-1: f(x) = -1

now we evaluate the slope of the interval (-4, -1) -->

(dy / dx)
-1-59
----
-1-(-4)

=

-60/3 = -20

Therefore, f(h) at h=3 is -20

Now, with the following coordinates of f(h): (1,-28), (3,-20), we can evaluate the slope of f(h), i.e. the 'm' in y = mx+b -->
dy / dx --> (-20+28) / (3-1) = 8/2 = 4. Therefore, m = 4.

We now need to find the intercept term b to complete the formula y=mx+b. So substituting one of the two arbitrary h's (we take h=3) yields

-20 = 4*3 + b --> -20=12+b --> -32 = b. Therefore b = -32, and the average rate of change is y = 4h - 32.

'''

def big_intervals(int1, int2, int3, start_interval, stop_interval, arbitrary_pt1, arbitrary_pt2):
    x_arbitrary_pt1 = stop_interval+arbitrary_pt1
    x_arbitrary_pt2 = stop_interval+arbitrary_pt2
    #evaluation for the start point of the interval
    eval_start_interval = (int1*start_interval**2) + (int2*start_interval) + int3
    #evaluations for first arbitrary point
    eval_arbitrary_pt1 = (int1*x_arbitrary_pt1**2) + (int2*x_arbitrary_pt1) + int3
    slope1 = (eval_arbitrary_pt1 - eval_start_interval) / (x_arbitrary_pt1 - start_interval)
    #evaluations for second arbitrary point
    eval_arbitrary_pt2 = (int1*x_arbitrary_pt2**2) + (int2*x_arbitrary_pt2) + int3
    slope2 = (eval_arbitrary_pt2 - eval_start_interval) / (x_arbitrary_pt2 - start_interval)
    #find m
    m = (slope2 - slope1) / (arbitrary_pt2 - arbitrary_pt1)
    #find b
    b = (m*arbitrary_pt1)-slope1
    b = b/-1
    #return result
    print ("The average rate of change ( the function f(h) ) is: " + str(m) + "h" + "+" + str(b))


big_intervals(3,1,0,-3,-3,17,87)

