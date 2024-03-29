import numpy as np
import matplotlib.pyplot as plt

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
#(start point of the interval)].
#This sum is the area approximation.
#We parametrize the function with 1) the function expression (here 'xi' takes the place of 'x', that accounts for the iterative method
#of taking Riemann sums), 2) a, which is the x-value where the interval starts, 3) b, which is the x-value where the interval ends, and 4)
#the number of equal subdivisions (number of rectangles) we like to sum.
def integral_approx_left_riemann_sum(func, a, b, subdivs):
    dx=(b-a)/subdivs #evaluate the width of each rectangle
    i=0 #initialize i at 0 (for right Riemann sums i should start at 1)
    res=0
    while i < subdivs:
        xi=(dx*i+a) #function evaluation, see comment above
        res = res + dx*(eval(func)) #sum intermediate results
        print (res) #(optionally) output intermediate results
        i+=1 #increment i
    print (res) #output the final result, i.e. the approximate area under the curve, or the definite integral

#Example: f(x+3), interval 0 to 4, 4 subdivs
#with 1000 subdivisions (intervals) the result comes very close to the real result (=20) but never quite reaches it. 
#due to the left Riemann sum it always underestimates and will never converge
#Right Riemann sums work in reverse. They overestimate maximally when the number of subdivisions is small and get closer to the actual result
#when the number of subdivisions is high, but never converge to the actual result.

#*************FUNCTION CALL**************************
#integral_approx_left_riemann_sum('6*xi', -1, 0, 1000) #evaluate 'integral_approx_left_riemann_sum' with the function 'x+3'

#********************************************************************************************************************************************
#If, for example, the function which is integrated is the velocity function, then its (definite) integral is the net accumulation of change of the
#original function, i.e. the position function. In other words, the position function is the 'antiderivative' of the velocity function. 
#Taking the integral of the derivative recovers the original function.

#A definite integral (interval from a to b) of a function f(x) can be algebraically deduced from its antiderative F(x) by evaluating F(b) - F(a),
#where a is the start value of the interval, and b its end value.
#********************************************************************************************************************************************

#function x² --- interval 3 to 7 --- 4 equal subdivisions
#dx=1, a=3, 9+16+25+36=86

#function x+3 --- interval 3 to 7 --- 4 equal subdivisions
#dx=1, a=3, 6+7+8+9=30
#right Riemann sum: 7+8+9+10=34

#*********************************************************************************************************************************************
#Euler's method
#*********************************************************************************************************************************************

#Given a first-order ODE and a given initial value, Euler's method helps us to estimate the solution of that ODE. The result may be visualized
#with a slope field or a curve. We define a suitable 'delta_x' or step size to essentially define how narrow or wide a field we would like to
#estimate. The smaller the step size, the more accurate the estimate is going to become.

#The formula for deriving the 'new x' is: old x + delta_x.
#The formula for deriving the 'new y' is: old y + old dy/dx*delta_x

#For example, if we are given y'=y and an initial value of y(0) = 1, and step size 1, the results up to y(4) would look as follows:
#x: 0, y: 1, dy/dx: 1
#x: 1, y: 2, dy/dx: 2
#x: 2, y: 4, dy/dx: 4
#x: 3, y: 8, dy/dx: 8

#The parameters for below algorithm are:
# - f_prime, which is essentially the ODE (dy/dx=f_prime)
# - init_x, which is the x in the given initial value
# - init_y, which is the y in the given initial value
# - delta_x, which is the step size, i.e. how much is x nudged for each subsequent estimate
# - eval_limit, which is the last x value to be estimated
# -- optionally -- for the dict representation, we can choose an x to be evaluated (and correspondingly indexed into the dict)

def euler_method(f_prime, init_x, init_y, delta_x, eval_limit, eval_x):
    x=init_x
    y=init_y
    #result_list=[[x, y, eval(f_prime)]] #list representation
    result={x:(y, eval(f_prime))} #dict representation
    i=init_x
    while i < eval_limit:
        y=y+(eval(f_prime)*delta_x)
        x=x+delta_x
        #result_list.append([x, y, eval(f_prime)]) #list representation
        result[x]=(y, eval(f_prime)) #dict representation
        i+=delta_x
    #print (result[eval_x])
    #copy the dict entries into two plottable lists and plot the function estimate
    x_values, y_values=[],[]
    for x_val in result.keys():
        x_values.append(x_val)
    for y_val in result.values():
        y_values.append(y_val[0])
    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print (result)
    
euler_method('x+y', 1, 2, 0.2, 4, 2) #function call for y'=x+y with initial given value y(1) = 2

#The analytical solution for y'=x+y is [c]e^x-x-1. For above initial value y(1) = 2, the full solution (adding in the constant [c]) is
#-(ex-4e^x+e)/e. Evaluating the correctness of y(1) = 2, we plug 1 into the ODE solution, yielding -(e - 4e + e)/e = approximately
#-(2.7 - 10.8 + 2.7)/2.7 = 2. Checking the ODE constraint y'=x+y, we take the derivative of -(ex-4e^x+e)/e, which is -(-4e^x+1+e)/e, 
#and plugging in 1, we get approximately 3. Setting this 3 equal to x+y, we get 3 = 1 + 2, which is true.

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

Important trigonometric and other indefinite integrals:
INDEFINITE INTEGRAL [1/x]dx --> ln|x| + C
INDEFINITE INTEGRAL [e^x]dx --> e^x + C
INDEFINITE INTEGRAL [sin(x)]dx --> -cos(x) + C
INDEFINITE INTEGRAL [cos(x)]dx --> sin(x) + C

Important trig identities:
sin²x + cos²x = 1
sin²x = 1/2*(1-cos(2x)) --> double angle identity

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
the velocity is negative. Velocity, on the other hand, has a magnitude, so it is essentially a vector, whereas speed is represented by a scalar
(pure magnitude/size, no direction).

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

u-substitution (integral calculus):
-----------------------------------

When we want to integrate a function that looks like its derivative was found by the chain rule, we can try u-substitution. For the example
2x*cos(x²) we implement the following steps:

1) find u and du --> u is the 'inner function', i.e. x². du is then its derivative (the g'(x) of the chain rule), 2x. Using the way of the integral
notation we find that 2x is actually '2x dx'

2) rearrange the integral in the form u*du and substitute in u --> cos(u) du

3) integrate in terms of x and substitute u back in --> sin(x²) dx

4) for the final result, we need to add the (possible) constant, so --> sin(x²) + C

5) to test, we can reverse this and find the derivative and yield 2x*cos(x²).

Further example: integrate sqrt(5x) (int sqrt(5x) dx)

1) identify u and du --> u is 5x, du is its derivative, i.e. 5 dx

2) substitute in u and du --> sqrt(u)*du/5. Since du = 5dx, dx = du/5

3) In order to separate our du again to later get our (notational) 'pure' dx back, we can use the integration property of scaling an integral.
So we put 1/5 as a multiplier in front of the integral, becoming 1/5 int sqrt(u) du

4) Now we can integrate u --> 1/5 * 2/3 (u)^3/2 + C (which is equivalent to u^3/2 / 3/2, with sqrt(u) being u^1/2 and then reverse power rule). The 'du'
now disappears, since it is pure notational. ALWAYS KEEP THE 'u' IN PARENTHESES, I.E. ISOLATE IT!

5) Substituting back in the real value of u and solving with respect to x --> 2/15 (1/5 * 2/3) (5x)^3/2 + C becomes our result, which can be simplified
to 2/3*sqrt(5)x^3/2 + C

********************************************************************************************************************************************
If the derivative (the du) is not directly found in the function to be integrated, it needs to be 'made up' and multiplied by the expression
in order to get the usual 'chain rule format'. In this case, as shown in the above sqrt(5x) example, it is necessary to divide the derivative again.
This is why dx != du, but dx = du/5. We have to divide by the same '5' we added as du into the expression. If we didn't do that division, the whole
function expression would be different, in other words, the function would be 5*sqrt(5x) instead of sqrt(5x)
********************************************************************************************************************************************

Integration by parts
--------------------

We integrate 'by parts' if we take the integral a composite of two functions whose derivative was found by the product rule. This means that
we regard one of the functions as the derivative of another function. It can be any of the two functions to be integrated as a composite.
To find the integral of f(x)g'(x), we solve f(x)g(x) - the integral of f'(x)g(x). 
In order to find appropriate candidates for f(x) and g(x), respectively, it is important we set f(x) to be the function whose derivative is 
'easier' to take than for g(x). This way the second part of above formula (the integral of f'(x)g(x)) is relatively easy to solve for.

It might be necessary to use integration by parts twice, since the integral of the second part of the formula might not be obvious.

Example: int x cos(x) -->
set x to be f(x)
set cos(x) to be g'(x)

Then, x*cos(x) dx = x*sin(x) - (int 1*sin(x)...the latter being -cos(x)
The result, therefore, is x sin(x) + cos(x) + C

Another example -->

int x³*ln(x) dx

We set:
f(x) = ln(x)
g'(x) = x³

Using the formula f(x)g'(x) = f(x)g(x) - int (f'(x)g(x)) yields:
ln(x)*1/4x^4 - int ((1/x)*(1/4x^4)) --> result of the integral x^4/16
int x³*ln(x) dx = (ln(x)*(1/4x^4) - (x^4/16) + C

solving for a definite integral with upper bound e and lower bound 1 results in:
3e^4/16 + 1/16

Solving integrals with partial fractions
----------------------------------------

When we have two separate expressions, e.g. (x+2)(x-1) in the denominator of a function, in order to integrate, we need to split the function
in two parts, one with a variable 'A' in the nominator, the other with a variable 'B' in the nominator. Hence, if we have a function

1 - 2x
------
(x+1)(x-2)

we split it up into

A / x+1 --> x+1 being the first partial fraction, and
B / x-2 --> x-2 being the second partial fraction

Then we solve for both A and B by setting the nominator of the original function equal to A and B, respectively, diagonally times the respective
partial fractions. So in above case this would be 1 - 2x = A*x-2 + B*x+1.

We solve for A and B by setting x so that the respective other variable is set to 0.

Having calculated A and B, we substitute them back in and solve for the integral, so

int 1-2x / (x+1)(x-2) = int A / x+1 + int B / x-2.

Some examples -->

EXAMPLE 1:

int (1) / (x+1)(3x+1)

A / x+1 + B / 3x+1

1 = A(3x+1) + B(x+1)

substituting x = -1, A = -1/2
substituting x = -1/3, B = 3/2

-1/2    3/2
----  + --- 
x+1     3x+1

in the second fraction, to get to 3 in the nominator,
we need to multiply by 3. That means before the fraction,
we need to divide by 3 (or multiply by 1/3). Hence, the resulting integral is

-1/2 (ln(abs(x+1))) + (3/2 * 1/3) which is 1/2 (ln(abs(3x+1)))

-----------------------------------------------

EXAMPLE 2:

int (18-12x) / (4x-1)(x-4)

A / 4x-1 + B / x-4

18-12x = A(x-4) + B(4x-1)

substituting x = 1/4 --> A = -4
substituting x = 4 --> B = -2

-4          -2
--    +     --
4x-1        x-4

in the first fraction, to get to 4 in the nominator,
we need to multiply by 4. That means before the fraction,
we need to divide by 4 (or multiply by 1/4). Hence, the resulting integral is

(-4*1/4) which is -1 (ln(abs(4x+1))) - 2 (ln(abs(x-4)))

-------------------------------------------------------------

EXAMPLE 3:

int (1) / (x+2)(x-2)

A / x+2 + B / x-2

1 = A(x-2) + B(x+2)

substituting x = -2 --> A = -1/4
substituting x = 2 --> B = 1/4

-1/4     1/4
----  +  ---
x+2      x-2

We do not need to multiply anything since both fractions are now already
in the form '1/x' for which the indefinite integral is ln(abs(x)). Hence, the resulting integral is

-1/4 (ln(abs(x+2))) + 1/4 (ln(abs(x-2)))

-----------------------------------------------------------------

EXAMPLE 4:

int (2-4x) / (x+2)(x-3)

A / x+2 + B / x-3

2-4x = A(x-3) + B(x+2)

substituting x = -2 --> A = -2
substituting x = 3 --> B = -2

-2       -2
--   +   --
x+2      x-3

We do not need to multiply anything since both fractions are now already
in the form '1/x' for which the indefinite integral is ln(abs(x)). Hence, the resulting integral is

-2 (ln(abs(x+2))) -2 (ln(abs(x-3)))

---------------------------------------------------------------------------

SOLVING EXPONENTIAL AND LOGISTIC MODEL DIFFERENTIAL EQUATIONS
-------------------------------------------------------------

GENERAL SOLUTION FOR **EXPONENTIAL** MODELS

From rate of change to accumulation of change (from derivative to integral)

dP/dt = kP --> price change is proportional to a constant multiplied by the price
dP = kP dt --> rearrange in order to take the integral
dP/kP = dt
ln(P)/k = t + C --> take integral
ln(P) = kt + C
e^(ln(P) = e^kt+C
P = e^kt * e^C --> e^C is a constant and can be any arbitrary real number, hence 
P = C*e^kt

EXAMPLE:

A computer's price decreases proportionally to its initial price.
The initial price was USD 850. After two years, the price is now USD 306.
What will be the price after 5 years?

Following above general solution, we calculate

C=850 --> given initial price (price upon purchase)
calculate ratio of the given value after two years and initial price
P(t+2) = 306
------------  =  9/25
P(t)   = 850

which can be expressed as:

C*e^kt+2
--------   =  9/25
C*e^kt
rearrange:
C*e^kt+2k-kt = 9/25 
e^2k = 9/25
ln(e^2k) = ln(9/25)
2k = ln(9/25)
k = ln(9/25)/2

solve for the computer's price after 5 years, i.e. P(t) when t is 5
P(5) = 850*e^(ln(9/25)/2*5) --> this formula represents C*e^kt .. where C=850, k is ln(9/25)/2, and t is 5.

GENERAL SOLUTION FOR **LOGISTIC** MODELS

The difference between exponential and logistic models is that the latter takes into account a maximum 'carrying capacity' of whatever is modeled
(a population, for instance). It follows the Malthusian idea that a population cannot grow forever but has its natural limits. Whereas exponential
growth can grow to infinitely, logistic growth is 'capped', and if the maximum 'carrying capacity' is reached, the growth becomes 0.

As a concrete example, let's imagine a bottle which carries 1 liter. If, by a given method, you pour liquid in this bottle, the liquid volume inside
the bottle cannot grow any more if the bottle is full, i.e. it cannot hold more than 1 liter. The growth stops there. It reaches its natural limit
(because the bottle can only carry 1 liter).

The exponential formula *** dN/dt = rN *** is, for logistic equations, multiplied by the term 1 - N/K, where K is the carrying capacity. The closer N
(which is the 'population', or whatever else we model) is to K, the slower growth becomes until it stops completely, i.e. when N = K. With a small N,
growth is essentially exponential, however.

So, dN/dt = rN * 1-N/K
and
N(t) = N_init*K / N_init + (K-N_init)e^-rt, where N_init is the initial 'population', K is the carrying capacity, and N is the population.

If we have a bottle that carries a maximum of 1 liter (so K = 1), and an initial liquid level of 100 ml (N_init = 0.1), and furthermore the
observation that after 2 seconds of pouring a certain liquid into the bottle, the quantity doubles, we may now model the quantity
(which is essentially the net accumulation of change, the integral value) of that liquid after 5 seconds.

First we solve for r, which is the ratio between our initial qty and the experimental measurement after 2 seconds have elapsed.

QTY(t+2)
--------  =  2
QTY(t)

Following the examples above (exponential models) for finding k (which is now named r), we establish that r = ln(2)/2

We can now directly plug in all known values and solve for N(5) -->
N(5) = 0.1*1 / 0.1 + (1-0.1)*e^-(ln(2)/2*5) = approximately 0.39.
This means that after 5 seconds the qty in the bottle will be around 400 ml. Our 'cap' guarantees that the qty can never be higher than 1 liter.

'''








