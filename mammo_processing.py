import numpy as np
from PIL import Image
import cv2 
import satsense.features.lacunarity as lac

################################################################################

def mammo_flip_to_point_right(arr):
# Checks breast orientation based on the mean of the pixels
# in the left and right side of the image, then flips image accordingly.

    image = np.copy(arr)
    height, width = image.shape
    left_side_mean  = np.mean(image[ :, :int(width/2) ])
    right_side_mean = np.mean(image[ :, int(width/2): ])
    if (left_side_mean < right_side_mean):
        return np.flip(image, axis=1)
    else: 
        return image

################################################################################

def mammo_mask_flip_to_point_right(arr, mask):
# Checks breast orientation based on the mean of the pixels
# in the left and right side of the image, then flips image and
# mask accordingly.

    image = np.copy(arr)
    height, width = image.shape
    left_side_mean  = np.mean(image[ :, :int(width/2) ])
    right_side_mean = np.mean(image[ :, int(width/2): ])
    if (left_side_mean < right_side_mean):
        return (np.flip(image, axis=1), np.flip(mask, axis=1))
    else: 
        return (image,mask)

################################################################################

def crop_sides(arr, x_proportion=0.01, y_proportion=0.02):  
# Crop array sides by a given proportion. 
# The crop is applied at both extremes of the array, thus a 1% x_proportion
# results in a 2% total crop of the image width.

    height,width=arr.shape
    x0 = int( np.trunc(width*x_proportion     ) )
    x1 = int( np.trunc(width*(1-x_proportion) ) ) 
    y0 = int( np.trunc(height*y_proportion    ) )
    y1 = int( np.trunc(height*(1-y_proportion)) )
    return arr[y0:y1, x0:x1]        

################################################################################
     
def darken_sides(arr, x_proportion=0.01, y_proportion=0.02):  
# Change array sides to zero by a given proportion.
# The darkening is applied at both extremes of the array, thus a 1% x_proportion 
# results in a 2% total darkening of the image width.

    aux = arr.copy()
    height,width=arr.shape
    x0 = int( np.trunc(width*x_proportion     ) )
    x1 = int( np.trunc(width*(1-x_proportion) ) )
    y0 = int( np.trunc(height*y_proportion    ) )
    y1 = int( np.trunc(height*(1-y_proportion)) )
    aux[  :y0 ,   :  ] = 0
    aux[y1:   ,   :  ] = 0
    aux[  :   ,   :x0] = 0
    aux[  :   , x1:  ] = 0
    return aux

################################################################################

def binarize_with_threshold(arr, threshold=0.5, normalized=True):  
# Binarize array with given threshold.
# The result can be returned in the range [0,255] or [0,1], depending on
# the "normalized" parameter.

    if normalized:
        return np.where(arr>threshold, 1, 0)
    else:
        return np.where(arr>threshold, 255, 0)

################################################################################

def apply_mask(image, mask):       
# Apply given binary mask (in array format) to target array.
# Both the image and mask MUST have the same dimension. 

    return np.multiply(image,mask)

################################################################################

def generate_apply_mask(arr, threshold=0.5, normalized=True):   
# Generate a binary mask from the given array based on binarization with a given 
# threshold, then apply it to said array.

    return apply_mask(arr, binarize_with_threshold(arr,threshold, normalized))

################################################################################

def generate_black_background(arr):
# Generate a 'zeros' array with the same shape as the input array.

    return np.zeros(arr.shape)

################################################################################

def normalize(arr):
# Normalize array to the [0,1] range.

    if (np.max(arr)>1):
        return arr/255
    else:
        return arr

################################################################################

def get_largest_contour_index(contours):
# Get the index of the largest contour (based on area)
# from a given OpenCV contour array.

# Note: This code could be optimized to avoid using a 'for' cycle.

    max_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>max_area:
            max_area_index = i
            max_area = area
    return max_area_index

################################################################################

def largest_contour_segmenting_based_in_pixel_luminancy_threshold(image_array, bin_threshold=20, width_darken = 0.01, height_darken=0.03, normalized=False):
# Binarize the input array with a given threshold and obtain retain only the 
# largest countour. The rest of the image is darkened.

# Parameters:
#    image_array   : Mammogram image in numpy array format.
#    bin_threshold : Pixel value for image binarization. Change this value 
#                    depending on image luminancy (brightness).
#    width_darken  : Percentage (at each side of the image) that will be set to
#                    black. Aids in contour detection.
#    height_darken : Percentage (at each side of the image) that will be set to 
#                    black. Aids in contour detection.
#    normalized    : Whether the input (and output) image is normalized or not.

    #Darken image sides to ensure proper contour recognition (set sides to 0).
    image_dark_sides = darken_sides(image_array, x_proportion=width_darken, y_proportion=height_darken)  
    #Binarize image for contour recognition.
    image_bin = binarize_with_threshold(image_dark_sides, bin_threshold, normalized)   
    # Identify binarized image contours.
    [contours, contour_hierarchy] = cv2.findContours(image_bin, mode=cv2.RETR_CCOMP , method=cv2.CHAIN_APPROX_NONE)  
    # Get the largest contour index.
    largest_contour_index = get_largest_contour_index(contours)
    # Generate a black base image for contour with the same shape as the passed image.
    largest_contour_mask = generate_black_background(image_dark_sides)  
    # Generate a binary mask based on the largest contour.
    cv2.drawContours(image=largest_contour_mask, contours=[contours[largest_contour_index]], contourIdx=-1, color=(255), thickness=-1)
    # Obtain segmented image.
    image_segmented = apply_mask(image_array, largest_contour_mask)

    # Return segmented image normalized with respect to the darkened image.
    # This ensures the resulting luminancy histogram is as similar as possible 
    # to the original one.
    return  image_segmented * np.max(image_dark_sides)/np.max(image_segmented)

################################################################################

def crop_pectoral(arr_segmented, threshold=70, pectoral_angle=70, scanning_start_width = 0.02, scanning_start_height=0.04, normalized=False):
# This function darkens the pectoral region from the mammogram.
# Breast tissue from the input image must be previously segmented with 
# largest_contour_segmenting_based_in_pixel_luminancy_threshold().

#    Parameters:
#    arr_segmented         : Input image in array 'L' format.
#    pectoral_threshold    : Threshold for pectoral muscle detection. Increase this 
#                            value if the tissue-muscle contrast is low. Pectoral 
#                            segmentation performance is of course diminished for 
#                            low contrast mammograms.
#    pectoral_angle        : Angle of the pectoral muscle in the mammogram, used 
#                            when creating the pectoral crop line.
#    scanning_start_width  : Width value to start looking for end of pectoral. 
#                            Should be higher than the darkened percentage used 
#                            in largest_countour_seg...().
#    scanning_start_height : Height value to start looking for end of pectoral. 
#                            Should be higher than the darkened percentage used 
#                            in largest_countour_seg...().
#    normalized            : Whether the input (and output) image is normalized or 
#                            not.

    muscle_identification_mask = binarize_with_threshold(arr_segmented, threshold, normalized)
    height,width = muscle_identification_mask.shape

    x_coord = int(np.trunc(width  * scanning_start_width ))
    y_coord = int(np.trunc(height * scanning_start_height))

    # Search for the X value where the mask changes value, as this is where the
    # muscle tissue ends.
    while(muscle_identification_mask[y_coord,x_coord]!=0):
        x_coord = x_coord+1
    
    y_coord_inf = int(np.trunc( x_coord*np.tan(np.pi*pectoral_angle/180)      + y_coord ))
    x_coord_sup = int(np.trunc( y_coord*np.tan(np.pi*(90-pectoral_angle)/180) + x_coord ))

    point_1 = (0          ,           0)
    point_2 = (0          , y_coord_inf)
    point_3 = (x_coord_sup, 0          )

    pectoral_contour = np.array( [point_1, point_2, point_3] )

    # Generate a black base image for contour with the same shape as the passed image.
    pectoral_mask = generate_black_background(muscle_identification_mask)  
    # Generate a binary mask based on the largest contour.
    cv2.drawContours(image=pectoral_mask, contours=[pectoral_contour], contourIdx=-1, color=(255), thickness=-1)
    # Como quiero ELIMINAR la region de la mascara en lugar de conservarla, debo invertirla antes de aplicarla.
    inverted_pectoral_mask=np.copy(pectoral_mask)
    inverted_pectoral_mask[np.where(pectoral_mask==0  )] = 255
    inverted_pectoral_mask[np.where(pectoral_mask==255)] = 0

    image_without_pectoral = apply_mask(arr_segmented, inverted_pectoral_mask)

    # Return segmented image normalized with respect to the darkened image.
    # This ensures the resulting luminancy histogram is as similar as possible to the original one.
    return image_without_pectoral*np.max(arr_segmented)/np.max(image_without_pectoral)

################################################################################


def expand_image_borders(arr, expanded_height=3481, expanded_width=2746, resize_percentage=0.3):
# This function rezises the image array by the indicated percentage, then fills
# with zeros up to the expanded width and height.
# The method only resizes the image if the resized height and width are less than 
# the expanded height and width.
# Parameters: 
#    arr              : image in array format
#    expanded_height  : output image array height
#    expanded_width   : output image array width
#    resize_percentage: percentage by which the image array is resized (resized 
#                       image dimensions are 1+resize_percentage)

    image_expanded = np.zeros((expanded_height,expanded_width)) # Generate expanded background
    image_height, image_width = arr.shape                       # Obtain original shape.
    # Resize shape if possible, then "paste" image in the background.
    if (image_height*(1+resize_percentage)<expanded_height)and(image_width*(1+resize_percentage)<expanded_width):
        arr2=cv2.resize(arr, dsize=(int(image_width*(1+resize_percentage)), int(image_height*(1+resize_percentage))), interpolation=cv2.INTER_NEAREST)
        image_expanded[:arr2.shape[0], :arr2.shape[1]] = arr2
    # If resize exceeds desired dimensions, only "paste" image in background.
    else:
        image_expanded[:image_height, :image_width] = arr
    return image_expanded

################################################################################

def fill_mask(mask_bin):
#Returns the largest contour of the input mask filled.

    try:
        [contours, contour_hierarchy] = cv2.findContours(mask_bin, mode=cv2.RETR_CCOMP , method=cv2.CHAIN_APPROX_NONE)  
        # Get the largest contour index.
        largest_contour_index = get_largest_contour_index(contours)
        # Generate a black base image for contour with the same shape as the passed image.
        largest_contour_mask = generate_black_background(mask_bin)  
        cv2.drawContours(image=largest_contour_mask, contours=[contours[largest_contour_index]], contourIdx=-1, color=(255), thickness=-1)
        return largest_contour_mask
    except:
        return mask_bin

################################################################################

def complete_mammo_mask_preprocessing(arr, mask_arr, contour_threshold=20, width_darken = 0.01, height_darken=0.03, pectoral_threshold=70, pectoral_angle=70, expanded_height=3481, expanded_width=2746, resize_percentage=0.3, mask_threshold = 0, normalized=False):
# Performs complete mammogram segmentation. Rotates image if needed, identifies 
# and segments breast tissue and eliminates the pectoral muscle and other labels. 
# Then resizes image and expands borders to desired shape. The expanded borders 
# beyond the resized image are filled with zeros. 
# If the resized image shape would exceed the desired expanded shape, image is not 
# resized. 
# Additionally, this function returns the mask binarized (and flipped when 
# necessary). The mask contour can also be filled (enabled by default). 
# Default parameters are set for my specific DDSM dataset.
# Parameters:
#    arr                : Input image in array 'L' format.
#    mask_arr           : Input mask in array format.
#    contour_threshold  : Threshold for contour binarization and detection. 
#                        Choose a higher number if background noise is high, or 
#                         a lower number to conserve a smoother surface.
#    width_darken       : Percentage (at each side of the image) that will be set 
#                         to black (0). Aids in contour detection. Increase this
#                         value if the image presents bright zones around the 
#                         edges.
#    height_darken      : Percentage (at each side of the image) that will be set 
#                         to black (0). Aids in contour detection. Increase this 
#                         value if the image presents bright zones around the 
#                         edges.
#    pectoral_threshold : Threshold for pectoral muscle detection. Increase this 
#                         value if the tissue-muscle contrast is low. Pectoral 
#                         segmentation performance is of course diminished for 
#                         low contrast mammograms.
#    pectoral_angle     : Angle of the pectoral muscle in the mammogram, used 
#                         when creating the pectoral crop line.
#    expanded_height    : output image array height.
#    expanded_width     : output image array width.
#    resize_percentage  : percentage by which the image array is resized (resized 
#                         image dimensions are 1+resize_percentage).
#    mask_threshold     : threshold by which the mask is binarized.

    mammo_flipped, mask_flipped       = mammo_mask_flip_to_point_right(arr, mask_arr)
    segmented_breast                  = largest_contour_segmenting_based_in_pixel_luminancy_threshold(mammo_flipped, contour_threshold, width_darken, height_darken, normalized)
    segmented_breast_without_pectoral = expand_image_borders(crop_pectoral(segmented_breast, threshold=pectoral_threshold, pectoral_angle=pectoral_angle), expanded_height=expanded_height, expanded_width=expanded_width, resize_percentage=resize_percentage)    
    
    binarized_mask = binarize_with_threshold(mask_flipped, mask_threshold)
    filled_mask    = fill_mask(binarized_mask)
    expanded_mask  = expand_image_borders(filled_mask, expanded_height=expanded_height, expanded_width=expanded_width, resize_percentage=resize_percentage)
    
    return (segmented_breast_without_pectoral, expanded_mask)

################################################################################

def fractal_dimension(Z, threshold=0.5):
# This code is implemented here, but obtained from the following link:
# https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1


    # Only for 2d image
    assert(len(Z.shape) == 2)
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform Z into a binary array
    Z = (Z < threshold)
    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

################################################################################

def generate_fractal_dimension_map(arr, fractal_window=11, binarization_threshold = 0.5):
 # Compute fractal dimension map for a square numpy array.
    fractal_dim_map = np.zeros(arr.shape)
    side_padding    = int(fractal_window/2)

    for y_index in range(len(fractal_dim_map) -fractal_window):
        for x_index in range(len(fractal_dim_map) -fractal_window):
            fractal_dim_map[y_index +side_padding, x_index +side_padding] = fractal_dimension(arr[y_index:y_index+fractal_window-1, x_index:x_index+fractal_window-1], threshold=binarization_threshold)

    return fractal_dim_map

################################################################################

def eliminate_most_common_element(arr):
# Used for cleaning fractal dimension map background.    
    new_arr = np.copy(arr)
    u, c = np.unique(new_arr, return_counts=True)
    most_common_pixel = u[c.argmax()]
    new_arr = np.where(new_arr == most_common_pixel, 0, new_arr)
    return new_arr

################################################################################

def generate_lacunarity_dimension_map(arr, lacunarity_window=32, lacunarity_boxes=(8,8)):
# Compute lacunarity map for a square numpy array.
    lacunarity_dim_map = np.zeros(arr.shape)
    side_padding    = int(lacunarity_window/2)

    for y_index in range(len(lacunarity_dim_map) -lacunarity_window):
        for x_index in range(len(lacunarity_dim_map) -lacunarity_window):
            snippet = arr[y_index:y_index+lacunarity_window-1, x_index:x_index+lacunarity_window-1]
            lacunarity_dim_map[y_index +side_padding, x_index +side_padding] = lac.Lacunarity.compute(snippet, lacunarity_boxes)[0]

    return lacunarity_dim_map

################################################################################

def apply_clahe_norm(img, clip=10.0, tile=(8, 8)):

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img
