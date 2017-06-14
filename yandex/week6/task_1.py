from skimage.io import imread
from skimage.util import img_as_float
from sklearn.cluster import KMeans
import matplotlib.pylab as plb
import pylab




def main():
  image = imread('parrots.jpg')
  img_as_flt = img_as_float(image)
  print(img_as_flt)
  plb.imshow(image)
  pylab.show()
main()
