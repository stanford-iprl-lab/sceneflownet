import numpy as np
from math import ceil,trunc,floor,sin,cos,atan,acos,sqrt
import numpy
import sys
from mayavi import mlab as mayalab

#def rotmatrix_angleaxis(rot):
#  angleaxis = np.zeros((3,)) 
#  angleaxis[0] = rot[2,1] - rot[1,2]
#  angleaxis[1] = rot[0,2] - rot[2,0]
#  angleaxis[2] = rot[1,0] - rot[0,1]
#  angleaxis = angleaxis / (np.linalg.norm(angleaxis) + 0.000001)
#  tmp = (rot[0,0] + rot[1,1] + rot[2,2] - 1) * 0.5
#  if tmp > 1.0:
#    tmp = 1.0
#  elif tmp < -1.0:
#    tmp = -1.0
#  angle = np.arccos( tmp )
#  angleaxis *= angle
#  assert(np.all(np.logical_not(np.isnan(angleaxis))))
#  return angleaxis


#def angleaxis_rotmatrix(angleaxis):
#  angle = np.linalg.norm(angleaxis)
#  axis = angleaxis / (angle + 0.000001)
#  c = np.cos(angle)
#  v = 1 - c
#  s = np.sin(angle)
#  rot = np.zeros((3,3))
#  rot[0,0] = axis[0] ** 2 * v + c
#  rot[0,1] = axis[0] * axis[1] * v - axis[2] * s
#  rot[0,2] = axis[0] * axis[2] * v + axis[1] * s
#  rot[1,0] = axis[0] * axis[1] * v + axis[2] * s
#  rot[1,1] = axis[1] ** 2 * v + c
#  rot[1,2] = axis[1] * axis[2] * v - axis[0] * s
#  rot[2,0] = axis[0] * axis[2] * v - axis[1] * s
#  rot[2,1] = axis[1] * axis[2] * v + axis[0] * s
#  rot[2,2] = axis[2] ** 2 * v + c
#  return rot



def quaternion_matrix(quaternion):
  _EPS = np.finfo(float).eps * 4.0
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)
  if n < _EPS:
    return np.identity(4)
  q *= sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def angle_between_vectors(r1,r2):
  return acos(np.sum(r1*r2)/np.linalg.norm(r1)/np.linalg.norm(r2))

def quaternion_multiply(quaternion1, quaternion0):
  w0, x0, y0, z0 = quaternion0
  w1, x1, y1, z1 = quaternion1
  return numpy.array([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=numpy.float64)


def quaternion_from_matrix(matrix,isprecise=False):
  M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
  if isprecise:
    q = numpy.empty((4, ))
    t = numpy.trace(M)
    if t > M[3, 3]:
      q[0] = t
      q[3] = M[1, 0] - M[0, 1]
      q[2] = M[0, 2] - M[2, 0]
      q[1] = M[2, 1] - M[1, 2]
    else:
      i, j, k = 0, 1, 2
      if M[1, 1] > M[0, 0]:
        i, j, k = 1, 2, 0
      if M[2, 2] > M[i, i]: 
        i, j, k = 2, 0, 1
      t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
      q[i] = t
      q[j] = M[i, j] + M[j, i]
      q[k] = M[k, i] + M[i, k]
      q[3] = M[k, j] - M[j, k]
      q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
  else:
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = numpy.linalg.eigh(K)
    q = V[[3, 0, 1, 2], numpy.argmax(w)]
  if q[0] < 0.0:
    numpy.negative(q, q)
  return q


def quaternion_rotation(quater,r):
  w1, x1, y1, z1 = quater
  if r.ndim == 1:
    x2, y2, z2  = r
  else:
    x2, y2, z2 = np.squeeze(np.split(r,3,axis=-1))
  
  wm =         - x1 * x2 - y1 * y2 - z1 * z2
  xm = w1 * x2           + y1 * z2 - z1 * y2
  ym = w1 * y2           + z1 * x2 - x1 * z2
  zm = w1 * z2           + x1 * y2 - y1 * x2

  x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
  y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
  z = -wm * z1 + zm * w1 - xm * y1 + ym * x1

  return np.stack((x,y,z),axis=-1)

def quaternion_decompose(quater,r1):
  r1 = r1 / np.linalg.norm(r1)
  m1,n1,p1 = r1
  q0,qx,qy,qz = quater
  alpha = 2 * atan( (m1 * qx + n1 * qy + p1 * qz) / q0 )
  r2 = quaternion_rotation(quater,r1)
  theta = angle_between_vectors(r1,r2)
  quater1 = np.array([cos(alpha/2),sin(alpha/2)*m1,sin(alpha/2)*n1,sin(alpha/2)*p1])
  q10,q1x,q1y,q1z = quater1
  q30 = cos(theta/2) 
  r3 = np.cross(r1,r2)
  r3 = r3/np.linalg.norm(r3)
  q3x,q3y,q3z = r3 * sin(theta/2)
  quater3 = np.array([q30,q3x,q3y,q3z])
  return quater3,quater1

def angle_axis_from_quaternion(quater):
  angle = 2 * acos(quater[0])
  axis = quater[1:4]/sin(angle/2)
  return angle, axis

def quaternion_from_angle_axis(angle,axis):
  q0 = cos(angle/2)
  qx,qy,qz = axis * sin(angle/2)
  return np.array([q0,qx,qy,qz])

def quaternion_shrink(quater,R,Cn):
  Cn = float(Cn)
  quater3,quater1 = quaternion_decompose(quater,R)
  angle1, axis1 = angle_axis_from_quaternion(quater1)
  rotation_module = 2 * np.pi / Cn
  print('%f %f' % (rotation_module/2,angle1))
  if angle1 > rotation_module/2:
    angle1 = angle1 - ceil((angle1-rotation_module/2)/ rotation_module) * rotation_module
    print('new %f %f' % (rotation_module/2,angle1)) 
  elif angle1 < -rotation_module/2:
    angle1 = angle1 - floor((angle1+rotation_module/2)/ rotation_module) * rotation_module
    print('new %f %f' % (rotation_module/2,angle1))
  quater1_new = quaternion_from_angle_axis(angle1,axis1)
  quater_final = quaternion_multiply(quater3,quater1_new) 
  return quater_final,quater3 

if __name__ == '__main__':
  q = np.array([0.0,0.1073,0.0521,0.2411])
  q /= np.linalg.norm(q)
  quater = q#quaternion_multiply(q3,q1)
  Rsym = np.array([0.001752,-0.999992,-0.003552]) 
  sys.path.append('/home/lins/pygeometry/obj_codes') 
  from objloader import obj_loader   
  v,f,vn = obj_loader('/home/lins/model.obj') 
  tran_o = np.array([0.1,0.1,0.])
  v = v + tran_o
  v_o = v
  p20 = quaternion_rotation(quater,v_o)
  print(p20.shape)
  quater_s,quater_v = quaternion_shrink(quater,Rsym,100)
  p80 = quaternion_rotation(quater_v,v_o)
  tran1 = quaternion_rotation(quater_v,tran_o)
  tran2 = quaternion_rotation(quater,tran_o)
  p80 = p80 + tran2 - tran1
  mayalab.points3d(v[:,0],v[:,1],v[:,2],color=(1,0,0),mode='point') 
  mayalab.points3d(p20[:,0],p20[:,1],p20[:,2],color=(0,1,0),mode='point')
  mayalab.points3d(p80[:,0],p80[:,1],p80[:,2],color=(0,0,1),mode='point')
  mayalab.show()


