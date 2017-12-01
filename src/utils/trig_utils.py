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


# def two_points_angle(pt1, pt2):
#     return np.arccos(np.dot(pt1,pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ang1 = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    ang2 = 2 * np.pi - ang1
    return min(ang1, ang2)


def point_in_cylinder(pt1, pt2, points, radius_sq):
    from scipy.spatial.distance import cdist
    dist = np.linalg.norm(pt1-pt2)
    elc_ori = (pt2-pt1) / dist # norm(elc_ori)=1mm
    elc_line = np.array([pt1 + elc_ori*t for t in np.linspace(0, dist, 100)])
    dists = np.min(cdist(elc_line, points), 0)
    # inside = dists <= radius_sq
    return np.where(dists <= radius_sq)[0]


def point_in_cylinder2(pt1, pt2, testpt, radius_sq):
    # Name: CylTest_CapsFirst
    # Orig: Greg James - gjames@NVIDIA.com
    # Lisc: Free code - no warranty & no money back.  Use it all you want
    #
    # This function tests if the 3D point 'testpt' lies within an arbitrarily
    # oriented cylinder. The cylinder is defined by an axis from 'pt1' to 'pt2',
    # the axis having a length squared of 'lengthsq' (pre-compute for each cylinder
    # to avoid repeated work!), and radius squared of 'radius_sq'.
    #    The function tests against the end caps first, which is cheap -> only
    # a single dot product to test against the parallel cylinder caps.  If the
    # point is within these, more work is done to find the distance of the point
    # from the cylinder axis.
    #    Fancy Math (TM) makes the whole test possible with only two dot-products
    # a subtract, and two multiplies.  For clarity, the 2nd mult is kept as a
    # divide.  It might be faster to change this to a mult by also passing in
    # 1/lengthsq and using that instead.
    #    Elminiate the first 3 subtracts by specifying the cylinder as a base
    # point on one end cap and a vector to the other end cap (pass in {dx,dy,dz}
    # instead of 'pt2' ).
    #
    # The dot product is constant along a plane perpendicular to a vector.
    # The magnitude of the cross product divided by one vector length is
    # constant along a cylinder surface defined by the other vector as axis.

    lengthsq = np.linalg.norm(pt2-pt1)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dz = pt2[2] - pt1[2]
    pdx = testpt[0] - pt1[0]
    pdy = testpt[1] - pt1[1]
    pdz = testpt[2] - pt1[2]

    # Dot the d and pd vectors to see if point lies behind the
    # cylinder cap at pt1.x, pt1.y, pt1.z
    dot = pdx * dx + pdy * dy + pdz * dz

    # If dot is less than zero the point is behind the pt1 cap.
    # If greater than the cylinder axis line segment length squared
    # then the point is outside the other end cap at pt2.
    if dot < 0.0 or dot > lengthsq:
        return False
    else:
        # Point lies within the parallel caps, so find
        # distance squared from point to line, using the fact that sin^2 + cos^2 = 1
        # the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
        # Carefull: '*' means mult for scalars and dotproduct for vectors
        # In short, where dist is pt distance to cyl axis:
        # dist = sin( pd to d ) * |pd|
        # distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
        # dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
        # dsq = pd * pd - dot * dot / lengthsq
        #  where lengthsq is d*d or |d|^2 that is passed into this function

        # distance squared to the cylinder axis:
        dsq = abs(pdx*pdx + pdy*pdy + pdz*pdz - dot*dot/lengthsq)
        return dsq < radius_sq

