import math
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from trianglesolver import solve, degree
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data
X, Y, Z = axes3d.get_test_data(0.05)


# Plot basic wireframe
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# plt.show()


# We can use the wireframe to plot how a plane transforms after sphere inversion
z_plane = deepcopy(X)


# Need to map inversion to each point on z_plane
# fig = plt.figure()
# ax = plt.axes(projection='3d')

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
xline = x
yline = y


def calc_distance(p1, p2):
    """
    Calculate Euclidean distance between any two points for any number of dimensions
    Input points as lists with coordinates as elements like [x,y,z...]
    """
    dist_by_dim = [(p1[i] - p2[i]) for i in range(len(p1))]
    squared_terms = [dim_dist**2 for dim_dist in dist_by_dim]
    total_dist = np.sqrt(sum(squared_terms))
    return total_dist


# Check that distance of anywhere on the unit circle is 1
# print(calc_distance([0, 0], [x[33], y[33]]))

# Create a 2D vertical line at x=1
xline = np.linspace(1, 1, 101)
yline = np.linspace(-10, 10, 101)


def calc_slope(point):
    """
    Calculate the slope of the line from a point and the origin in 2D
    """
    return point[1] / point[0]


def circl_invert(coords):
    """
    Do a 2D circle inversion of a set of coordinates on the unit circle:
    - Calculate the distance from that point to the origin of the circle
    - Find the slope of the line which the point and the origin lie on
    - Find the angle between that line and the x-axis
    - Solve the triangle using hypotenuse length and the two angles to get the inverse coordinates
    """
    inverse = [[], []]
    for point in coords:
        new_x = 1 / point[0]
        new_y = 1 / point[1]
        # Still need to adjust x and y based on the ray they should be on
        inverse[0].append(new_x)
        inverse[1].append(new_y)
    return inverse


a,b,c,A,B,C = solve(a=3,b=4,C=(math.pi/2))
print(a,b,c,A,B,C)
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

