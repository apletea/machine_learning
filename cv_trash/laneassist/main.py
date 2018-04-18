import cv2
import os
import numpy as np
#and not i < -(W / H) *j + H
#not j > (float(3)/5*W / H) *i + 7*H/8
import pylab
import matplotlib.pyplot as plt

x = [7, 1200, 600, 642]
y = [719, 719, 319, 319]
X = [400, 800, 400, 800]
Y = [719, 719, 0, 0]

src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) )
dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) )

M = cv2.getPerspectiveTransform(src, dst)


def warpimage(img,M):
  img_size = img.shape
  wapr = cv2.warpPerspective(img, M, (1280,720), flags=cv2.INTER_NEAREST)
  return wapr

vc = cv2.VideoCapture('./first.mp4')
H,W =720,1280
mask = np.zeros((H,W))
for i in range(0,H):
  for j in range(0,W):
    if  not j > (float(3)/5*W / H) *i + 7*H/8 and not i < -(W / H) *j + H:
      mask[i, j] = 1


i=0
while True:
  _,frame = vc.read()
  H,W = frame.shape[0],frame.shape[1]
  blur = cv2.blur(frame,(5,5))
  
  edges = cv2.Canny(blur,30,120)
#  edges = mask * edges
  lines = cv2.HoughLines(edges,2,np.pi/60,200)
  for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
#  edges = warpimage(edges,M)
  cv2.imshow('frame',frame)
  cv2.waitKey(5)
  i+=1
 
