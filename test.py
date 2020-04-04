import math
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
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
res = np.array(map(circle_inversion, z_plane))
# for p in np.nditer(z_plane):
#     print(p**2)
print(res)
quit()

ax.plot_wireframe(X, Y, z_plane, rstride=10, cstride=10)
plt.show()

inverted_z = circle_inversion(0, 0, z_plane)
print(inverted_z)

ax.plot_wireframe(X, Y, inverted_z, rstride=10, cstride=10)
plt.show()

