import numpy as np

from skimage import data, img_as_float, img_as_ubyte
from skimage.segmentation import chan_vese
from skimage.color import rgb2gray
from skimage.io import imsave, imread
from skimage.transform import resize

from multiprocessing import Pool

def transform_training(x, shape=(224,224), nprocess=6):
    dev = np.asarray([resize(i, shape, 0) for i in x])

    if dev.dtype == "float64" and dev.max() <= 1.0:
        dev = (dev*255).astype(np.uint8)

    segmented = None
    for i in range(0, len(dev), nprocess):
        with Pool(processes=nprocess) as pool:
            batch = pool.map(segment_ChanVese, dev[i:i+nprocess])
        batch = np.asarray(batch)

        segmented = batch if segmented is None else np.vstack((segmented, batch))
    
    return segmented

def segment_ChanVese(source, max_iter = 200, show_plot = False, max_size = 1024):

    image = img_as_float(source)
    image_grayscale = rgb2gray(image)
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(image_grayscale, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=max_iter,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)

    mask = cv[0]
    mask_multiple = np.zeros(source.shape)

    # finding coordinates to set 0
    center = (mask.shape[0]//2, mask.shape[1]//2)

    # take 2/5 of the pixels in the center, 1/5 left + 1/5 right
    pixels_x, pixels_y = int(0.15 * mask.shape[0]), int(0.15 * mask.shape[1])

    # take the values and the x,y coordinates in the center
    centers = [(mask[i][j], (i,j)) for i in range(center[0]-pixels_x, center[0]+pixels_x)\
                          for j in range(center[1]-pixels_y, center[1]+pixels_y)]
    # find the most represented in the center
    centers_values, counts_centers = np.unique(np.asarray([i[0] for i in centers]), return_counts=True)

    # the central value is the most represented
    central_value = centers_values[counts_centers.argmax()]

    cluster = set()
    # add all the pixels matching the same center value
    to_analyze = set([i[1] for i in centers if i[0] == central_value])
    while len(to_analyze) != 0:
        curr = to_analyze.pop()
        cluster.add(curr)
        mask_multiple[curr[0]][curr[1]][:] = 1
        news_neig = [(i,j) for i in [curr[0]-1, curr[0], curr[0]+1] for j in [curr[1]-1, curr[1], curr[1]+1]\
                        if i > -1 and i < mask.shape[0] and j > -1 and j < mask.shape[1]\
                        and mask[i][j] == central_value and ((i,j) not in cluster) and ((i,j) not in to_analyze)]
        to_analyze = to_analyze.union(set(news_neig))

    # find original mask
    mask_original = np.zeros(source.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == central_value:
                mask_original[i][j][:] = 1
   
    source_mask = source * mask_multiple
    source_mask = source_mask.astype("uint8")

    return source_mask