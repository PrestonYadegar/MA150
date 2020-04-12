import math
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data
X, Y, Z = axes3d.get_test_data(0.05)


# Plot basic wireframe
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# plt.show()


# We can use the wireframe to plot how a plane transforms after sphere inversion
z_plane = deepcopy(X)


def circle_inversion(radius, center, geo_object):
    """
    Give the radius and center of the circle/sphere to invert by,
    geo_object is a line, plane, or some geometric object targeted for inversion
    """
    return geo_object**2


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


def distance(p1, p2, dim=2):
    """
    Calculate the distance between any two points given the dimensions of the space
    """
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))


# Check that distance of anywhere on the unit circle is 1
# print(distance([0, 0], [x[10], y[10]]))

# Create a vertical line at x=1
xline = np.linspace(1, 1, 101)
yline = np.linspace(-10, 10, 101)


def invert(coords, circle):
    """
    Do a 2D circle inversion of a set of coordinates on the unit circle
    """
    inverse = [[], []]
    for point in coords:
        new_x = 1 / point[0]
        new_y = 1 / point[1]
        # Still need to adjust x and y based on the ray they should be on
        inverse[0].append(new_x)
        inverse[1].append(new_y)
    return inverse

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
p1 = [0.25, 0.25]
p1_dist = distance([0, 0], p1)
dist_ratio = (1 / p1_dist) / p1_dist
p2 = [p1[0] * dist_ratio, p1[1] * dist_ratio]
p2_dist = distance([0, 0], p2)
print(p1)
print(p1_dist)
print(p2)
print(p2_dist)
print((p1_dist * p2_dist) == 1)
quit()

plt.plot(xline, yline)
print(xline)
plt.show()
quit()

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# plt.show()

inverted_z = circle_inversion(0, 0, z_plane)
print(inverted_z)

ax.plot_wireframe(X, Y, inverted_z, rstride=10, cstride=10)
plt.show()

