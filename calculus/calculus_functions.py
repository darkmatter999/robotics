import numpy as np

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
    #calculate the slopes (derivatives) for both x1 and x2. Due to the tiny size of the nudge parameter, the difference between x_nudged and x is 
    #of course almost zero, but not quite zero. It is technically the limit as 'slope_x' goes to zero.
    slope_x1 = (eval_x1_nudged - eval_x1) / ((x1+nudge) - x1)
    print (slope_x1)
    slope_x2 = (eval_x2_nudged - eval_x2) / ((x2+nudge) - x2)
    print (slope_x2)
    #find m and b and construct the general slope formula y=mx+b
    m = (slope_x2 - slope_x1) / (x2-x1)
    b = (m*x1) - slope_x1
    b = b/-1

    print ("The estimated derivative ( f'(x) ) for any arbitrary x-values is: " + str(m) + "x" + "+" + str(b))

#derivative_estimation(3,1,0,1,2,0.00000000001)

#possible extension of above derivative_estimation algorithm for any kind of polynomial. The 'int's' are now just put inside a list and iterated over.
intlist = [3,1,0]
i = 0
j = len(intlist)
x = 2
eval_x = 0
while i < len(intlist)-1 and j > 0:
    eval_step = (intlist[i]*x**(j-1))
    #print (eval_step)
    eval_x = eval_x + eval_step
    i = i+1
    j = j-1
eval_x = eval_x + intlist[-1]
#print (eval_x)

#approximating the definite integral (area under the curve) with a left rectangular Riemann sum.
#We divide the area into n equal subdivisions and define the function interval for which the area approximation is required.
#We take the sum of [0 to dx (interval) - 1] dx (intervals) * f(xi) where xi is dx*[function evaluation at running variable i + a
# (start point of the interval)].
#This sum is the area approximation.
def integral_approx_left_riemann_sum(func, a, b, subdivs):
    dx=(b-a)/subdivs
    i=0
    res=0
    while i < subdivs:
        res = res + eval(func)
        print (res)
        i = i+1
    print (res)

#Example: f(x+3), interval 0 to 4, 4 subdivs
#with 1000 subdivisions (intervals) the result comes very close to the real result (=20) but never quite reaches it. 
#due to the left Riemann sum it always underestimates and will never converge
#Right Riemann sums work in reverse. They overestimate maximally when the number of subdivisions is small and get closer to the actual result
#when the number of subdivisions is high, but never converge to the actual result.
integral_approx_left_riemann_sum('dx*(dx*i+a+3)', 3, 7, 10000)

#function x² --- interval 3 to 7 --- 4 equal subdivisions
#dx=1, a=3, 9+16+25+36=86

#function x+3 --- interval 3 to 7 --- 4 equal subdivisions
#dx=1, a=3, 6+7+8+9=30
#right Riemann sum: 7+8+9+10=34

#1.5+1.75+....



'''
Important trigonometric and other derivatives:
d/dx [ln(x)] --> 1/x
d/dx [e^x] -->e^x
d/dx [sin(x)] --> cos(x)
d/dx [cos(x)] --> -sin(x)
d/dx [tan(x)] --> sec²(x)
d/dx [cot(x)] --> -csc²(x)
d/dx [sec(x)] --> tan(x)*sec(x)
d/dx [csc(x)] --> -(cot(x)(csc(x))
d/dx [a^x] --> ln(a)*a^x
d/dx [loga(x)] --> 1/(ln(a)*x)
d/dx [asin(x)] --> 1/sqrt(1-x²)
d/dx [acos(x)] --> -(1/sqrt(1-x²))
d/dx [atan(x)] --> 1/1+x²

Important rules:
----------------

Chain rule --> f'(g(x)) * g'(x)
The chain rule is used when differentiating a composite function such as sin(3x² + x) or (x+1)³. 
f(x) is considered as the 'outer function' (sin(x) in above first example) and g(x) as the 'inner function' (3x² + x in above first example).
You take the derivative of the outer function with respect to the inner function and multiply this with the derivative of the inner function.

Derivatives of inverse functions --> h'(x) = 1/f'(h(x))
If f = x² + 3x, then h is the inverse of it. If f(2) = 5, then h(5) = 2. Using above equation, h'(5) is then 1/f'(2), i.e. the 2 replacing h(x).
Meanwhile, f'(2) can be evaluated from f', i.e. the derivative of f.

Position/velocity/speed/acceleration:
-------------------------------------

Speed is the magnitude, or absolute value of velocity. This means it's possible to 'speed up' even when 'moving backward', i.e. even when 
the velocity is negative.

Mean value theorem (MVT):
-------------------------

The MVT guarantees, for a function *f* that's differentiable over an interval from *a* to *b*, that there exists a number *c* on that interval
such that f'(c), i.e. slope of tangent line, is equal to the function's average rate of change (slope of secant line) over that interval -->
f'(c) = f(b) - f(a) / b - a

In order for the MVT to be applicable, *f* must be differentiable over the open interval (a,b) and continuous over the closed interval [a,b].

Connecting f, f' and f'' (finding extreme points and inflections):
------------------------------------------------------------------

1) Setting the first derivative to 0 (or to points where f is undefined) yields the extreme points of a function, i.e. the points where the slope is 0. 
These can be relative or absolute minimum or maximum points. To find out if a given extremum is a minimum or a maximum, the extremum must be plugged 
into the second derivative. This 'second derivative test' tells us that if negative, we have a maximum point - if positive, we have a minimum.

2) This latter finding is due to concavity. If the function has a 'cap' form, it is concave downwards. The 'tip' of the cap is the maximum point, i.e.
from there on it can only go downwards. Contrary to that, a 'cup' form means the function has a concave upwards. The bottom of the cup is the minimum,
i.e. from there on it can only go upwards.

3) The first derivative is negative when the function is decreasing. Likewise, the first derivative is positive when the function is increasing.

4) Setting the second derivative (f'') to 0 (or to points where f'' is undefined) yields **possible** inflection points, i.e. points where the 
concavity changes from up to down or vice versa. In order to check if an extremum of f'' (i.e. a point where the slope of f' is 0) is actually 
an inflection point, we need to check all relevant function intervals (e.g. -inf to first inflection candidate, first inflection candidate to +inf) 
for sign change. If the sign changes for the 'next' interval, e.g. if a negative evaluation of f'' becomes positive, then we have an inflection point.

Parametric equations:
---------------------

Parametric equations are eq.s of x and y with respect to something else, such as time.
The differentiation rules of a set of parametric equations are (dy/dt)/(dx/dt)
'''








