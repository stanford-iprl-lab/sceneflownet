import math 
import sys
import os
import numpy as np
from numpy.linalg import inv
from math import *

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    '''
       Args:
           dist :         distance from camera position to the origin
           azimuth_deg:   azimuth degree in 360 degresses starting from positive x axis
           elevation_deg: elevation_deg in 360 degresses
       Returns:
           (x,y,z) :      coordinates in xyz 
    '''
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    '''
       Calculate quaternion from yaw, pitch, roll
       Args:
            yaw: rotation degress along z axis
            pitch: rotation degrees along y axis
            roll: rotation degress along x axis
       Return:
            quaternion
    '''
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy):
    '''
       qx * qy
       first rotation qy and then rotate qx
    '''
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return (q1, q2, q3, q4)


def IoU(x, y):
    # segmentation IoU of 2 binary map
    return 1.0 * np.logical_and(x, y).sum() / np.logical_or(x, y).sum()

def APWithIoU(x, y, thres, min_portion=0.001):
    # segmentation Average Precision with min IoU thres
    # x and y are id maps, x is ground truth, y is prediction
    # x and y are of shape (H, W), x[i, j] / y[i, j] is the gt/pred class at pixel [i, j]
    # IoU should be permutation invariant
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    tp, total_p = 0, 0
    area = np.prod(y.shape)
    for y_ in y_unique:
        if (y == y_).sum() * 1.0 / area < min_portion:
            #print("skip!")
            continue
        total_p += 1
        for x_ in x_unique:
            if IoU(x == x_, y == y_) >= thres:
                tp += 1
                break
    #print("number of predictions: %d" % total_p)
    #total_p = len(y_unique)
    return tp * 1.0 / (total_p + 1e-5)

def mAP(x, y, thres_list):
    return np.mean([APWithIoU(x, y, thres) for thres in thres_list])


if __name__ == "__main__":
    x = np.array([
        [1, 1, 1, 1, 2, 2],
        [1, 1, 2, 2, 2, 2],
        [1, 3, 3, 2, 2, 2],
        [3, 3, 3, 3, 3, 3]
        ])
    y = np.array([
        [3, 3, 7, 7, 7, 7],
        [3, 3, 3, 7, 7, 7],
        [3, 5, 5, 5, 7, 7],
        [5, 5, 5, 5, 5, 5]
        ])
    thres_list = np.linspace(0, 1, 21)
    for thres in np.linspace(0, 1, 21):
        print(APWithIoU(x, y, thres))
    print(mAP(x, y, thres_list))
