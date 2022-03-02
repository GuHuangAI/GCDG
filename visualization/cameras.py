"""A camera library for projection."""
import json

import numpy as np

try:
    import google3
    from google3.pyglib import gfile
    from cvx2 import latest as cv2
    GOOGLE3 = True
except:
    import cv2
    GOOGLE3 = False


class Camera():
  """A simple camera class."""

  def __init__(self,
               rvec=np.zeros(3),
               tvec=np.zeros(3),
               matrix=np.eye(3),
               dist=np.zeros(5),
               name=None,
               size=(None, None)):
    """Camera constructor.

    Args:
      rvec: The rotation vector of the camera in Rodrigues' rotation formula.
        An `np.array` with the shape of `(3,)`.
      tvec: The translation vector of the camera. An `np.array` with the
        shape of `(3,)`.
      matrix: The intrinsic of the camera. An `np.array` with the
        shape of `(3,)`.
      dist: The distorion coefficients (k1, k2, p1, p2, k3). An `np.array` with
        the shape of `(5,)`.
      name: A string indicates the name of this camera.
      size: A tuple `(width, height)` indicates the size of the captured image.
    """
    self.rvec = rvec
    self.tvec = tvec
    self.matrix = matrix
    self.dist = dist
    self.name = name
    self.size = size

  def project(self, points3d):
    """Project a set of 3D points to 2D for this camera.

    This function uses cv2.projectPoints to calculate 2D points from 3D points.
    Args:
      points3d: The 3D point coordinates in the world system. An `np.array`
        with the shape of `(N1, N2, ..., Nm, 3)`.
    Returns:
      The 2D point coordinates in the image plane. An `np.array` with the
        shape of `(N1, N2, ..., Nm, 2)`.
    """
    shape = points3d.shape
    points3d = points3d.reshape(-1, 1, 3)
    points2d, _ = cv2.projectPoints(points3d, self.rvec, self.tvec,
                                    self.matrix.astype('float64'),
                                    self.dist.astype('float64'))
    return points2d.reshape(*shape[:-1], 2)


class CameraGroup():
  """A simple multi camera class."""

  def __init__(self, cams=None):
    self.cams = cams

  def project(self, points3d):
    """Project a set of 3D points to 2D for all cameras.

    Args:
      points3d: The 3D point coordinates in the world system. An `np.array`
        with the shape of `(N1, N2, ..., Nm, 3)`.
    Returns:
      The 2D point coordinates in the image plane. An `np.array` with the
        shape of `(#cams, N1, N2, ..., Nm, 2)`.
    """
    points2d = np.stack([cam.project(points3d) for cam in self.cams])
    return points2d

  @classmethod
  def load(cls, path):
    """Load data from json file."""
    if '/cns/' in path:
      with gfile.Open(path, 'r') as f:
        data = json.load(f)
    else:
      with open(path, 'r') as f:
        data = json.load(f)

    keys = sorted(data.keys())
    cams = []
    for key in keys:
      if key == 'metadata':
        continue

      else:
        name = data[key]['name']
        size = data[key]['size']
        matrix = np.array(data[key]['matrix'])
        dist = np.array(data[key]['distortions'])
        rvec = np.array(data[key]['rotation'])
        tvec = np.array(data[key]['translation'])

      cams.append(Camera(rvec, tvec, matrix, dist, name, size))

    return cls(cams)
