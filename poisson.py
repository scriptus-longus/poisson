#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.sparse import lil_matrix, linalg

IMG_SOURCE = ""
IMG_TARGET = ""
MASK       = ""

class edit(object):
  def adj(self, p):
    #return adj pts
    return [(p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)] 

  def poissonMatrix(self, n, pts):
    ret = lil_matrix((n,n))
    ret.setdiag(4)   
    p = pts.tolist()

    # find A
    for i in range(n):
      for a in self.adj(p[i]):
        if list(a) in p:
          j = p.index(list(a))
          # set -1
          ret[i,j] = -1 
          pass     
 
    return ret 

  def process(self, source, target, mask, offset=(0,0)):
    # find channels
    channel = 3 
    n = np.count_nonzero(mask == 0)    

    ret = target
    # list of pts in mask
    x,y = np.where(mask == 0)
    pts = np.concatenate((x.reshape((x.shape[0], 1)), y.reshape((y.shape[0], 1))), axis=1)
     
    A = self.poissonMatrix(n, pts)

    b = np.zeros((n))
    # calculate b
    for c in range(channel):
      for i, p in enumerate(pts):
        # define shortcuts
        s = source
        
        # compute laplacian 
        b[i] = 4*s[p[0],p[1],c] -s[p[0]-1,p[1],c] - s[p[0]+1,p[1],c] - s[p[0],p[1]-1,c] - s[p[0],p[1]+1,c]

        for a in self.adj(p):
          if mask[a[0], a[1]] == 1:
            b[i] += target[a[0]+offset[0], a[1]+offset[1], c]

      # solve Ax=b
      x = linalg.cg(A,b)
      ret = ret.astype(int)
      # fill ret x 
      for i, p in enumerate(pts):
        ret[p[0], p[1], c] = int(round(x[0][i]))
    return ret

if __name__ == "__main__":
  s = cv2.imread(IMG_SOURCE)
  t = cv2.imread(IMG_TARGET)
  mask = cv2.imread(MASK)

  H,W = s.shape[:2]
  # resize mask and s
  if W > 200:
    scaler = W/H
    H = int(round(200/scaler))
    W = 200

  mask = cv2.resize(mask, (W,H))
  s = cv2.resize(s, (W,H))

  # normalize mask(binary)
  mask = mask[:,:,0]
  mask = np.divide(mask, 255).round().astype(int)
 
  # start editing
  p = edit()
   
  img = np.zeros(t.shape, dtype=int)
 
  img = p.process(s, t, mask)

  plt.imshow(img)
  plt.show()
