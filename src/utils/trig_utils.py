import numpy as np


def unit_normal(a, b, c):
    # unit normal vector of plane defined by points a, b, and c
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return x / magnitude, y / magnitude, z / magnitude


def poly_area(poly):
    # area of polygon poly
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def perimeter(points):
    peri = dist3d(points[0], points[-1])
    for ind in range(len(points) - 1):
        peri += dist3d(points[ind], points[ind + 1])
    return peri


def points_dists(points):
    dists = [dist3d(points[0], points[-1])]
    for ind in range(len(points) - 1):
        dists.append(dist3d(points[ind], points[ind + 1]))
    return np.array(dists)


def dist3d(a, b):
    return np.linalg.norm(a-b)