import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad
import random

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR  HERE
    dx2 = dx**2
    dy2 = dy**2
    dxy = dx*dy
    
    Sx = convolve(dx2,window)
    Sy = convolve(dy2,window)
    Sxy= convolve(dxy,window)
    
    response = Sx * Sy - np.square(Sxy) - k * np.square(Sx + Sy)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    std = np.std(patch)
    mean = np.mean(patch)
    if std > 0.0:
        feature = (patch - mean) / std
    else:
        feature = patch - mean
    feature = feature.reshape(-1)
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)
    print dists.shape
    ### YOUR CODE HERE
    for i in range(N):
        dist = dists[i, :]
        if np.min(dist) / (np.partition(dist, 2)[1]) <= threshold:
            matches.append([i, np.argmin(dist)])
    matches = np.array(matches).reshape(-1, 2)
    ### END YOUR CODE
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    #print m
#    print c
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation
        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers
    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers
    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)
    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
    max_inliers = np.zeros(N,dtype='uint16')
    n_inliers = 0
    H = None
    # RANSAC iteration start

    ### YOUR CODE HERE

    for i in range(n_iters):
        n_inliers = 0
        current_inliers = np.zeros(N,dtype='uint16')
        # 1. Select random points
        random_indices = random.sample(range(matched1.shape[0]), n_samples)
        random_matched1 = [matched1[i] for i in random_indices]
        random_matched2 = [matched2[i] for i in random_indices]

        # 2. Compute affine transformation matrix
        affine_transformation_matrix, _, _, _ = np.linalg.lstsq(random_matched2, random_matched1)

        #3. Detected inliers
        for i in range(len(matched1)):
            if np.sum((matched1[i] - np.matmul(matched2[i],affine_transformation_matrix))**2) < threshold:
                current_inliers[i] = 1
                n_inliers += 1
        # 4. Keep the largest set of inliers
        if n_inliers > np.sum(max_inliers):
            max_inliers = np.copy(current_inliers)
            H = affine_transformation_matrix

    # 5. Re-compute least-squares estimate on all of the inliers

    affine_transformation_matrix, _, _, _ = np.linalg.lstsq(keypoints2[matches[max_inliers][:,1]],
                                                            keypoints1[matches[max_inliers][:,0]])
    ### END YOUR CODE
    return H, np.array([matches[i] for i in range(np.array(max_inliers).shape[0]) if max_inliers[i]==1])


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:
    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector
    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)
    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))


    ### YOUR CODE HERE
    # For each block - create a histogram
    for i in range(rows):
        for j in range(cols):
            for z in range(pixels_per_cell[0]):
                for t in range(pixels_per_cell[1]):
                    # Divide the cell's value between it's 2 closest bins
                    distance_to_lower_bin = theta_cells[i][j][z][t] % degrees_per_bin
                    division_ratio = float(distance_to_lower_bin) / degrees_per_bin
                    lower_bin_num = (int(np.trunc(theta_cells[i][j][z][t] / degrees_per_bin))) % 9

                    # Since the angles are circular - 0 comes after 8.

                    upper_bin_num = (lower_bin_num + 1) % 9
                    # print('value {}, lower bin {}'.format(theta_cells[i][j][z][t], lower_bin_num))
                    cells[i][j][upper_bin_num] += (division_ratio*G_cells[i][j][z][t])
                    # print(i,j,lower_bin_num)
                    cells[i][j][lower_bin_num] += ((1-division_ratio) * G_cells[i][j][z][t])

            # Normalize
            cells[i][j] = cells[i][j] / np.linalg.norm(cells[i][j])
    block = cells.flatten()
    ### YOUR CODE HERE
    
    return block
