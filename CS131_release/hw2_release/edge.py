import numpy as np
import cv2
def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    for i in xrange(0,Hi):
        for j in xrange(0,Wi):
            out[i][j] = np.sum(padded[i:i+Hk,j:j+Wk] * np.flip(kernel[:Hk][:Wk],axis=0))/(Hk*Wk)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))
    k = (size-1)/2
    ### YOUR CODE HERE
    for x in xrange(0,size):
        for y in xrange(0,size):
            top = (x-k)**2 + (y-k)**2
            bot = 2*(sigma**2)
            exp = np.exp(-float(top)/bot)
            kernel[x][y] = (1/((2*np.pi)*sigma**2))*exp
    ### END YOUR CODE
    print kernel
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """
    padded = np.pad(img, 1, mode='edge')
    out = np.zeros(img.shape)

    ### YOUR CODE HERE
    for i in xrange(1,padded.shape[0]-1):
        for j in xrange(1,padded.shape[1]-1):
            out[i-1][j-1] =  (padded[i][j+1] - padded[i][j-1])/float(2)
    ### END YOUR CODE
    ### END YOUR CODE
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """
    padded = np.pad(img, 1, mode='edge')
    out = np.zeros(img.shape)

    ### YOUR CODE HERE
    for i in xrange(1,padded.shape[0]-1):
        for j in xrange(1,padded.shape[1]-1):
            out[i-1][j-1] =  (padded[i+1][j] - padded[i-1][j])/float(2)
    ### END YOUR CODE
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx**2+Gx**2)
    theta= np.arctan2(Gx,Gy)
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    nms = np.copy(G)
    magnitude = G
    for i in range(theta.shape[0]-1):
        for j in range(theta.shape[1]-1):
            if (theta[i,j]<=22.5 or theta[i,j]>157.5):
                if(magnitude[i,j]<=magnitude[i-1,j]) and (magnitude[i,j]<=magnitude[i+1,j]): nms[i,j]=0
            if (theta[i,j]>22.5 and theta[i,j]<=67.5):
                if(magnitude[i,j]<=magnitude[i-1,j-1]) and (magnitude[i,j]<=magnitude[i+1,j+1]): nms[i,j]=0
            if (theta[i,j]>67.5 and theta[i,j]<=112.5):
                if(magnitude[i,j]<=magnitude[i+1,j+1]) and (magnitude[i,j]<=magnitude[i-1,j-1]): nms[i,j]=0
            if (theta[i,j]>112.5 and theta[i,j]<=157.5):
                if(magnitude[i,j]<=magnitude[i+1,j-1]) and (magnitude[i,j]<=magnitude[i-1,j+1]): nms[i,j]=0

    ### BEGIN YOUR CODE
#    print G
#    print theta
#    for x in xrange(1,H):
#        for y in xrange(1,W):
#            if theta[x][y] == 45:
#                if (G[x][y] != np.max(G[x-1][y-1],np.max(G[x][y],G[x+1][y+1]))):
#                    out[x][y] = 0
#                else:
#                    out[x][y] = G[x][y]
#            if theta[x][y] == 90:
##                if (G[x][y] != np.max(G[x][y-1],np.max(G[x][y],G[x][y+1]))):
 #                   out[x][y] = 0
 #               else:
 #                   out[x][y] = G[x][y]
 #           if theta[x][y] == 135:
  #              if (G[x][y] != np.max(G[x+1][y+1],np.max(G[x][y],G[x-1][y-1]))):
   #                 out[x][y] = 0
    #            else:
     #               out[x][y] = G[x][y]
      #      if theta[x][y] == 0:
       #         if (G[x][y] != np.max(G[x-1][y],np.max(G[x][y],G[x+1][y]))):
        #            out[x][y] = 0
         #       else:
          #          out[x][y] = G[x][y]
               
    ### END YOUR CODE

    return nms

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)
    img = img*25
    ### YOUR CODE HERE
    weak_edges = np.array(weak_edges,np.bool)
    strong_edges = np.array(strong_edges,np.bool)

    for x in xrange(0,img.shape[0]):
        for y in xrange(0,img.shape[1]):
            if img[x][y] > high :
                strong_edges[x][y] = True
            elif img[x][y] > low :
                weak_edges[x][y] = True
                ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    for i in xrange(H):
        for j in xrange(W):
            if (strong_edges[i][j]):
                neighbors = get_neighbors(i,j,H,W)
                edges[i][j] = True
                for neighbor in  neighbors:
                    if weak_edges[neighbor[0]][neighbor[1]]:
                        edges[neighbor[0]][neighbor[1]] = True
                        neighbors.append(get_neighbors(neighbor[0],neighbor[1],H,W)[0]) 
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size,sigma)
    smothed = conv(img,kernel)
    G,theta = gradient(smothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(len(x_idxs)):
    x = xs[i]
    y = ys[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
        rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
        accumulator[rho, t_idx] += 1
    ### END YOUR CODE

    return accumulator, rhos, thetas
