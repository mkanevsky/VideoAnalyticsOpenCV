import numpy as np
from scipy.spatial import distance as dist

def order_points(pts):
    """Order points in a clock-wise order, important for pose estimation """

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # sort left-most according to y coordinate
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # find bottom and top right
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")




def intersect_line(p11,p12,p21,p22):
    """Find line intersection, each line is defined by two points """
    xdiff = (p11[0] - p12[0], p21[0] - p22[0])
    ydiff = (p11[1] - p12[1], p21[1] - p22[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(p11,p12), det(p21,p22))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def makeLine(p1,p2):
    """Create (a,b) line representation (y=a*x + b) from two points"""
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    a = (y1 - y2) / (x1 - x2)
    b = (x1 * y2 - x2 * y1) / (x1 - x2)

    return a,b

def point_on_line(line, angles,pt, distance,direction_positive = False):
    import math

    """Find a point on a given line, with a certain distance from a source point"""
    if direction_positive and line[0] < 0:
        distance = -1*distance

    a = angles
    #a[0] = 180 - abs(a[0])
    a = [(angle * np.pi/180) for angle in a]


    #x = pt[0] + distance * np.cos(np.arctan(line[0]))*np.cos(angles[1])
    x = pt[0] + distance * np.cos(a[0]) * np.cos(a[1])
    y = line[0] * x  + line[1]

    return (int(x),int(y))

