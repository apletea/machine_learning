import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = ((np.array(image,np.float32) ) * 0.5) **2
    out = np.array(out,np.uint8)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    ### YOUR CODE HERE
    out = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in xrange(0,image.shape[0]):
        for j in xrange(0,image.shape[1]):
            out[i][j][0] = (image[i][j][0]/3 + image[i][j][1]/3 + image[i][j][2]/3)
            out[i][j][1] = out[i][j][0]
            out[i][j][2] = out[i][j][0]
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    change = np.zeros((image.shape[0],image.shape[1]))
    ### YOUR CODE HERE
    
    out = image.copy()
    out[:,:,channel] = change
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    g = image.copy()
    lab = color.rgb2lab(g)
    out = np.zeros((image.shape[0],image.shape[1],image.shape[2]))

    ### YOUR CODE HERE
    out[:,:,0] = lab [:,:,channel]
    out[:,:,1] = lab [:,:,channel]
    out[:,:,2] = lab [:,:,channel]
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = np.zeros((image.shape[0],image.shape[1],image.shape[2]))

    ### YOUR CODE HERE
    out[:,:,0] = hsv [:,:,channel]
    out[:,:,1] = hsv [:,:,channel]
    out[:,:,2] = hsv [:,:,channel]
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
