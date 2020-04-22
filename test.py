import math
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from trianglesolver import solve, degree, ssa, sas
from mpl_toolkits.mplot3d import axes3d

"""
Roadmap:
1) Define basic units (points, lines/shapes)
2) Define distance calculator
3) Inversion calculator in 2D, 3D 
4) Plotting/visualizations
"""

"""
Points are defined by lists of coordinates like [0,0,0] for the origin in 3D
Lines and other shapes are defined as lists of points rather than unique objects
This is a flexible (but lazy) approach for inversion (it's no more complicated to invert a point or line or circle)
"""


def calc_distance(p1, p2):
    """
    Calculate Euclidean distance between any two points for any number of dimensions
    Input points as lists with coordinates as elements like [x,y,z...]
    """
    dist_by_dim = [(p1[i] - p2[i]) for i in range(len(p1))]
    squared_terms = [dim_dist**2 for dim_dist in dist_by_dim]
    total_dist = np.sqrt(sum(squared_terms))
    return total_dist


def calc_inverse_dist(point, radius=1):
    """
    Calculate the distance for the inverse point from a circle inversion of given radius
    """
    original_distance = calc_distance([0, 0], point)
    inverse_distance = (radius**2) / original_distance
    return inverse_distance


# Check that distance of anywhere on the unit circle is 1
# print(calc_distance([0, 0], [x[33], y[33]]))


def calc_slope(point):
    """
    Calculate the slope of the line from a point and the origin in 2D
    """
    if point[0] != 0:
        return point[1] / point[0]
    else:
        # What slope to return when x-change is 0?
        # return np.inf * np.sign(point[1])
        # Let's just return a big value, but don't round off later
        # I would rather have some numerical error in just this case than
        # an rounding/truncating error term in every case
        return 99999999 * np.sign(point[1])


def line_solve(a=None, b=None, c=None, A=None, B=None, C=None):
    """
    When a side is 0, we are just dealing with a vertical/horizontal line calculation
    If one side is 0, the other sides are equal and all angles are 0
    """
    a = 0 if A == 0 else a
    b = 0 if B == 0 else b
    c = 0 if C == 0 else c
    if a == 0:
        b = b if b is not None else c
        c = c if c is not None else b
    elif b == 0:
        a = a if a is not None else c
        c = c if c is not None else a
    elif c == 0:
        a = a if a is not None else b
        b = b if b is not None else a
    return a, b, c, 0, 0, 0


def solve_all(a=None, b=None, c=None, A=None, B=None, C=None):
    """
    Lowercase letters represent sides, Uppercase letters are angle values for opposite sides
    """
    # If all sides and angles provided are positive
    if all(s > 0 for s in (a, b, c) if s is not None) and all(a != 0 for a in (A,B,C) if a is not None):
        # print('all positive')
        # Just use pre-built solve function
        a, b, c, A, B, C = solve(a, b, c, A, B, C)

    # If just one side is 0 or any angle is 0
    elif (sum(s == 0 for s in (a,b,c) if s is not None) == 1) or any(a == 0 for a in (A,B,C) if a is not None):
        # print('exactly one zero')
        # Use our function for the calculation
        a, b, c, A, B, C = line_solve(a, b, c, A, B, C)

    # If at least one side is negative
    elif sum(s < 0 for s in (a, b, c) if s is not None) < 3:
        # print('some negative')
        # Process the input and output to work with pre-built solve
        # Not done yet
        a, b, c, A, B, C = solve(a, b, c, A, B, C)
    else:
        raise ArithmeticError('Does not compute for provided input values')
    return a, b, c, A, B, C


def calc_point_inverse(point):
    """
    Do a 2D circle inversion of a set of coordinates on the unit circle:
    - Calculate the distance from that point to the origin of the circle
    - Find the slope of the line which the point and the origin lie on
    - Find the angle between that line and the x-axis
    - Use inverse hypotenuse length and angle to get coordinates of inverse point
    """
    point_slope = calc_slope(point)
    a, b, c, A, B, C = solve_all(a=1, b=point_slope, C=(math.pi/2))
    theta = deepcopy(B)
    inverse_distance = calc_inverse_dist(point)
    a, b, c, A, B, C = solve_all(c=inverse_distance, B=theta, C=(math.pi/2))
    inverse_point = [a, b]
    return inverse_point


# Currently only works for positive values b/c of triangle solver.
# Next steps: create custom triangle solver for the case in each quadrant
# Issues: we can only delegate cases where no angles or sides are zero
# We have to handle the case where a side/two angles are zero (e.g. vertical or horizontal line)
# Not all done yet
point = [0.1, 0]
# point = [0, 0.1]
inverse = calc_point_inverse(point)
print(point)
print(inverse)
quit()


def circle_invert(geo_object):
    """
    Run point inverse on all points in the object
    Object is just a list of points (which are themselves lists of coordinates)
    """
    inverse = []
    for point in geo_object:
        inverse.append(calc_point_inverse(point))
    return inverse


# print(circle_invert([[0.5, 0.5], [0.25, 0.25], [0.125, 0.125]]))
# quit()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Grab some test data
X, Y, Z = axes3d.get_test_data(0.05)


# Plot basic wireframe
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# plt.show()


# We can use the wireframe to plot how a plane transforms after sphere inversion
z_plane = deepcopy(X)


# Need to map inversion to each point on z_plane
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 10, 100)
xline = np.sin(zline)
yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'teal')
# plt.show()
# quit()

angles = np.linspace(0, 2, 100)
pi = np.pi
x = []
y = []
for angle in angles:
    x.append(np.cos(angle * pi))
    y.append(np.sin(angle * pi))
unit_circle_coordinates = [x, y]
circ_x = x
circ_y = y

# Create a 2D vertical line at x=1
xline = np.linspace(1, 1, 10000)
yline = np.linspace(0.0000001, 1000, 10000)
line = [[xline[i], yline[i]] for i in range(len(xline))]
inversion_result = circle_invert(line)
x,y = zip(*inversion_result)
# print(x)
# print(y)
plt.plot(circ_x,circ_y)
plt.plot(xline,yline)
plt.plot(x,y)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()
quit()

"""
|OP| * |OP’| = 1^2
|OP’| = 1 / |OP|
|OP| = 0.5
|OP’| = 2
O = (0,0)
P = (1, 1)
|OP| = distance(O, P) = sqrt(2)
|OP’| = 1 / sqrt(2)
|OP'| / |OP| = (1 / sqrt(2)) / sqrt(2) = 0.5
P’ = P * |OP'| / |OP| = P * 0.5 = (0.5, 0.5)

https://www.symbolab.com/solver/distance-calculator/
http://www.geometer.org/mathcircles/inversion.pdf
"""

# This code below will calculate the position for the inverse point but only for
# points on the line x=y
def invert_example():
    p1 = [0.25, 0.25]
    p1_dist = calc_distance([0, 0], p1)
    dist_ratio = (1 / p1_dist) / p1_dist
    p2 = [p1[0] * dist_ratio, p1[1] * dist_ratio]
    p2_dist = calc_distance([0, 0], p2)
    print(p1)
    print(p1_dist)
    print(p2)
    print(p2_dist)
    print((p1_dist * p2_dist) == 1)
    return None


p1 = [2, 2]
p1_dist = calc_distance([0, 0], p1)
print(p1_dist)
# invert_example()
quit()

# To find the inverse point, we need to get the ray for a point and the origin
# Then find the point on the line which satisfies the inversion distance equation
# If the point is located outside of the circle, then the inverse is inside
# So we can find (let's say) 1000 evenly spaced points in the circle and check which
# is closest to satisfying the inversion. We need to get the slope of the line to
# find the inverse point
# plt.plot(xline, yline)
print(xline)
# plt.show()
quit()

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# plt.show()

# inverted_z = circle_inversion(0, 0, z_plane)
# print(inverted_z)

# ax.plot_wireframe(X, Y, inverted_z, rstride=10, cstride=10)
# plt.show()

