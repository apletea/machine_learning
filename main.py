import cv2
import argparse
import numpy as np
import pandas as pd
from  mtcnn.mtcnn import MTCNN
import pydensecrf.densecrf as dcrf

from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray

def crf(original_image, mask_img):
    if (len(mask_img.shape) < 3):
        print('yep')
        mask_img = gray2rgb(mask_img)
        print(mask_img.shape)

    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)

    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape
            [0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)

    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))

def find_grad(img, keypoints):
    ans = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    kernel = np.ones((3,3),np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    left_h1,left_h2,left_w1,left_w2 = keypoints['left_eye'][1] + np.abs(int((keypoints['left_eye'][0] - keypoints['right_eye'][0]) * 0.1)) ,int((2*keypoints['left_eye'][1] + keypoints['mouth_left'][1])/3),keypoints['left_eye'][0] - int(np.abs(keypoints['left_eye'][0] -  keypoints['right_eye'][0]) * 0.25), keypoints['left_eye'][0] + int(np.abs(keypoints['left_eye'][0] -  keypoints['right_eye'][0]) * 0.25)
    right_h1,right_h2,right_w1, right_w2 =  keypoints['right_eye'][1] + np.abs(int((keypoints['left_eye'][0] - keypoints['right_eye'][0]) * 0.1)) ,int((2*keypoints['right_eye'][1] + keypoints['mouth_left'][1])/3),keypoints['right_eye'][0] - int(np.abs(keypoints['left_eye'][0] -  keypoints['right_eye'][0]) * 0.25), keypoints['right_eye'][0] + int(np.abs(keypoints['left_eye'][0] -  keypoints['right_eye'][0]) * 0.25)
    print ('right_h1 = {} right_h2={} right_w1={} right_w2={}'.format(str(right_h1),str(right_h2),str(right_w1),str(right_w2)))
    print ('left_h1 = {} left_h2 = {} left_w1={} left_w2={}'.format(str(left_h1),str(left_h2),str(left_w1),str(left_w2)))
    for i in range(left_w1, left_w2):
        a = [sobely[i][j] for j in range(left_h1, left_h2)]
        val = np.argmax(a)
        ans[left_h1:(left_h1 + val),i] = 1
        print (ans[left_h1:left_h1+val,i])
    for i in range(right_w1, right_w2):
        a = [sobely[i][j] for j in range(right_h1, right_h2)]
        val = np.argmax(a)
        ans[right_h1:(right_h1 + val),i] = 1
    cv2.imshow('return', ans)
    return ans

def apply_crf(img,mask):
    crf_output = crf(img, mask)
    return crf_output

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_img', action='store', dest='input_img', type=str, default='/home/apletea/Downloads/index.jpeg')
args = parser.parse_args()
detector  = MTCNN()
img = cv2.imread(args.input_img)
mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)
result = detector.detect_faces(img)
for i in range(len(result)):
    bounding_box = result[i]['box']
    keypoints = result[i]['keypoints']
    cv2.rectangle(img,(bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,155,255),
                                 2)
    cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
    
    mask = find_grad(img, keypoints)
#    mask = cv2.bitwise_or(mask,mask,find_grad(img,keypoints))
#   mask = mask | find_grad(img, keypoints)
mask = apply_crf(img, mask) 
print (sum(mask))
cv2.imshow("ivan_", img)
cv2.imshow('mask', np.array(mask,dtype=np.float32))
mask = np.array(mask , np.int) * 255
cv2.waitKey()
