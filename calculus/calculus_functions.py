#This program evaluates the average rate of change of a function at an arbitrary interval.
#It takes a parabolic function of the form '[int1]x^2+[int2]x+[int3]' (e.g. 5x^2-4x+3) as input
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

def rate_of_change_with_intervals(int1, int2, int3, start_interval, stop_interval, arbitrary_pt1, arbitrary_pt2):
    x_arbitrary_pt1 = stop_interval+arbitrary_pt1
    x_arbitrary_pt2 = stop_interval+arbitrary_pt2
    #evaluation for the start point of the interval
    eval_start_interval = (int1*start_interval**2) + (int2*start_interval) + int3
    #evaluations for first arbitrary point
    eval_arbitrary_pt1 = (int1*x_arbitrary_pt1**2) + (int2*x_arbitrary_pt1) + int3
    slope1 = (eval_arbitrary_pt1 - eval_start_interval) / (x_arbitrary_pt1 - start_interval)
    #if x at the arbitrary_pt1 is set very close to x (i.e. the start interval) then the slope1 represents 
    #a value very close to the derivative of f at start interval. This means that at 'extremely close' intervals the average rate
    #of change effectively equals the instantaneous rate of change. 
    print (slope1)
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


#rate_of_change_with_intervals(3,1,0,-3,0,-3.00009,-3.00002)


#A pretty accurate estimation of a derivative of a given function can be evaluated by following function.
#It takes a function, two x-values and a 'nudge' value as input parameters. The two x-values are just two different, arbitrary
#values at which the function is going to be evaluated. For both values, a respectively very close value (x + the very small 'nudge'
#parameter) is added and the respective secant lines (which are, for very close values, approximately the tangent lines) are evaluated.
#Finding the slopes of both x1 and x2 in this way, the approximate derivative f'(x) can be evaluated
def derivative_estimation(int1, int2, int3, x1, x2, nudge):
    #evaluate f(x1) and f(x2)
    eval_x1 = (int1*x1**2) + (int2*x1) + int3
    print (eval_x1)
    eval_x2 = (int1*x2**2) + (int2*x2) + int3
    print (eval_x2)
    #evaluate f(x1+the tiny nudge value) and f(x2+the tiny nudge value)
    eval_x1_nudged = (int1*(x1+nudge)**2) + (int2*(x1+nudge)) + int3
    print (eval_x1_nudged)
    eval_x2_nudged = (int1*(x2+nudge)**2) + (int2*(x2+nudge)) + int3
    print (eval_x2_nudged)
    #calculate the slopes for both x1 and x2. Due to the tiny size of the nudge parameter, the difference between x_nudged and x is of course
    #almost zero, but not quite zero. It is technically the limit as 'slope_x' goes to zero.
    slope_x1 = (eval_x1_nudged - eval_x1) / ((x1+nudge) - x1)
    print (slope_x1)
    slope_x2 = (eval_x2_nudged - eval_x2) / ((x2+nudge) - x2)
    print (slope_x2)
    #find m and b and construct the general slope formula y=mx+b
    m = (slope_x2 - slope_x1) / (x2-x1)
    b = (m*x1) - slope_x1
    b = b/-1

    print ("The estimated derivative ( f'(x) ) for any arbitrary x-values is: " + str(m) + "x" + "+" + str(b))

derivative_estimation(4,0,0,3,6,0.00000000001)

