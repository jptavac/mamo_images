
import numpy as np
from PIL import Image
import cv2 

################################################################################

def mammo_flip_to_point_right(arr):
# Checks breast orientation based on the mean of the pixels
# in the left and right side of the image, then flips accordingly.
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
    # in the left and right side of the image, then flips accordingly.
    image = np.copy(arr)
    height, width = image.shape
    left_side_mean  = np.mean(image[ :, :int(width/2) ])
    right_side_mean = np.mean(image[ :, int(width/2): ])
    if (left_side_mean < right_side_mean):
        return (np.flip(image, axis=1), np.flip(mask, axis=1))
    else: 
        return (image,mask)

################################################################################

def array_crop_sides(arr, x_proportion=0.01, y_proportion=0.02):  
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
     
def array_darken_sides(arr, x_proportion=0.01, y_proportion=0.02):  
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

def array_binarize_with_threshold(arr, threshold=125, normalized=False):  
# Binarize array with given threshold.
    if normalized==False:
        return np.where(arr>threshold, 255, 0)
    else:
        return np.where(arr>threshold, 1, 0)

################################################################################

def array_apply_mask(image, mask):       
# Apply given binary mask (in array format) to array.
    return np.multiply(image,mask)

################################################################################

def array_generate_apply_mask(arr, threshold=125):   
#Generate a binary mask from the given array with a given threshold, then apply
#it to said array.
    return array_apply_mask(arr, array_binarize_with_threshold(arr,threshold))

################################################################################

def array_generate_black_background(arr):
#Generate a 'zeros' array with the same shape as the parameter array.
    return np.zeros(arr.shape)

################################################################################

def array_normalize(arr):
#Normalize array to the [0,1] range.
    if (np.max(arr)>1):
        return arr/255
    else:
        return arr

################################################################################

def get_largest_contour_index(contours):
#Get the index of the largest contour (based on area)
#from the given OpenCV contour array.

    max_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area>max_area:
            max_area_index = i
            max_area = area
    return max_area_index

################################################################################

def largest_contour_segmenting_based_in_pixel_luminancy_threshold(image_array, bin_threshold=20, width_darken = 0.01, height_darken=0.03):
#Parameters description:
#    image_array   : Mammogram image in numpy array format
#    bin_threshold : Pixel value for image binarization. Change this value 
#                    depending on image luminancy (brightness).
#    width_darken  : Percentage (at each side of the image) that will be set to
#                    black. Aids in contour detection.
#    height_darken : Percentage (at each side of the image) that will be set to 
#                    black. Aids in contour detection.

    #Darken image sides to ensure proper contour recognition (set sides to 0).
    image_dark_sides = array_darken_sides(image_array, x_proportion=width_darken, y_proportion=height_darken)  
    #Binarize image for contour recognition.
    image_bin = array_binarize_with_threshold(image_dark_sides, bin_threshold)   
    # Identify binarized image contours.
    [contours, contour_hierarchy] = cv2.findContours(image_bin, mode=cv2.RETR_CCOMP , method=cv2.CHAIN_APPROX_NONE)  
    # Get the largest contour index.
    largest_contour_index = get_largest_contour_index(contours)
    # Generate a black base image for contour with the same shape as the passed image.
    largest_contour_mask = array_generate_black_background(image_dark_sides)  
    # Generate a binary mask based on the largest contour.
    cv2.drawContours(image=largest_contour_mask, contours=[contours[largest_contour_index]], contourIdx=-1, color=(255), thickness=-1)
    # Obtain segmented image.
    image_segmented = array_apply_mask(image_array, largest_contour_mask)

    # Return segmented image normalized with respect to the darkened image.
    # This ensures the resulting luminancy histogram is as similar as possible to the original one.
    return  image_segmented * np.max(image_dark_sides)/np.max(image_segmented)

################################################################################

def crop_pectoral( arr_segmented, threshold=70, pectoral_angle=70):
#This function crops (darkens) the pectoral region from the mammogram.
#    Breast tissue from the input image array must be segmented with the 
#    largest_contour_segmenting...() function.
#    Parameters:
#    arr_segmented      : Input image in array 'L' format.
#    pectoral_threshold : Threshold for pectoral muscle detection. Increase this 
#                         value if the tissue-muscle contrast is low. Pectoral 
#                         segmentation performance is of course diminished for 
#                         low contrast mammograms.
#    pectoral_angle     : Angle of the pectoral muscle in the mammogram, used 
#                         when creating the pectoral crop line.

    muscle_identification_mask = array_binarize_with_threshold(arr_segmented, threshold)
    height,width = muscle_identification_mask.shape
    #inicio el escaneo ya un poco adelantado en la imagen para reducir tiempo de 
    #procesamiento, debe ser como minimo mayor al porcentaje de oscurecimiento de ancho
    x_coord = int(np.trunc(width  *0.02))
    #configuro la altura como al 4% del borde superior
    y_coord = int(np.trunc(height *0.04))
    #busco donde la mascara cambia de valor (punto de la fila donde termina el musculo)
    while(muscle_identification_mask[y_coord,x_coord]!=0):
        x_coord = x_coord+1
    
    y_coord_inf = int(np.trunc( x_coord*np.tan(np.pi*pectoral_angle/180)      + y_coord ))
    x_coord_sup = int(np.trunc( y_coord*np.tan(np.pi*(90-pectoral_angle)/180) + x_coord ))

    point_1 = (0          ,           0)
    point_2 = (0          , y_coord_inf)
    point_3 = (x_coord_sup, 0          )

    pectoral_contour = np.array( [point_1, point_2, point_3] )

    # Generate a black base image for contour with the same shape as the passed image.
    pectoral_mask = array_generate_black_background(muscle_identification_mask)  
    # Generate a binary mask based on the largest contour.
    cv2.drawContours(image=pectoral_mask, contours=[pectoral_contour], contourIdx=-1, color=(255), thickness=-1)
    # Como quiero ELIMINAR la region de la mascara en lugar de conservarla, debo invertirla antes de aplicarla.
    inverted_pectoral_mask=np.copy(pectoral_mask)
    inverted_pectoral_mask[np.where(pectoral_mask==0  )] = 255
    inverted_pectoral_mask[np.where(pectoral_mask==255)] = 0

    image_without_pectoral = array_apply_mask(arr_segmented, inverted_pectoral_mask)
    # Return segmented image normalized with respect to the darkened image.
    # This ensures the resulting luminancy histogram is as similar as possible to the original one.
    return image_without_pectoral*np.max(arr_segmented)/np.max(image_without_pectoral)

################################################################################


def expand_image_borders(arr, expanded_height=3481, expanded_width=2746, resize_percentage=0.3):
#This function rezises the image array by the indicated percentage, then fills
#with zeros up to the expanded width and height.
#The method only resizes the image if the resized height and width are less than 
#the expanded height and width.
#Parameters: 
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
    # If resize exceeds desired dimensions, justo "paste" image in background.
    else:
        image_expanded[:image_height, :image_width] = arr
    return image_expanded

################################################################################

def complete_mammogram_preprocessing(arr, contour_threshold=20, width_darken = 0.01, height_darken=0.03, pectoral_threshold=70, pectoral_angle=70, expanded_height=3481, expanded_width=2746, resize_percentage=0.3):
#Performs complete mammogram segmentation. Rotates image if needed, identifies and segments breast tissue and eliminates the pectoral muscle and other labels. 
#Then resizes image and expands borders to desired shape. The expanded borders beyond the resized image are filled with zeros. If the resized
#image shape would exceed the desired expanded shape, image is not resized. 
#Default parameters are set for my specific DDSM dataset.
#Parameters:
#    arr                : Input image in array 'L' format.
#    contour_threshold  : Threshold for contour binarization and detection. Choose a higher number if background noise is high, or a lower number to conserve a smoother surface.
#    width_darken       : Percentage (at each side of the image) that will be set to black (0). Aids in contour detection. Increase this value if the image presents bright zones #around the edges.
#    height_darken      : Percentage (at each side of the image) that will be set to black (0). Aids in contour detection. Increase this value if the image presents bright zones #around the edges.
#    pectoral_threshold : Threshold for pectoral muscle detection. Increase this value if the tissue-muscle contrast is low. Pectoral segmentation performance is of course #diminished for low contrast mammograms.
#    pectoral_angle     : Angle of the pectoral muscle in the mammogram, used when creating the pectoral crop line.
#    expanded_height    : output image array height
#    expanded_width     : output image array width
#    resize_percentage  : percentage by which the image array is resized (resized image dimensions are 1+resize_percentage)
    segmented_breast                  = largest_contour_segmenting_based_in_pixel_luminancy_threshold(mammogram_flip_to_point_right(arr), contour_threshold, width_darken, height_darken)
    segmented_breast_without_pectoral = crop_pectoral(segmented_breast, threshold=pectoral_threshold, pectoral_angle=pectoral_angle)    
    return expand_image_borders(segmented_breast_without_pectoral, expanded_height=expanded_height, expanded_width=expanded_width, resize_percentage=resize_percentage)

################################################################################

def complete_mammo_mask_preprocessing(arr, mask_arr, contour_threshold=20, width_darken = 0.01, height_darken=0.03, pectoral_threshold=70, pectoral_angle=70, expanded_height=3481, expanded_width=2746, resize_percentage=0.3, mask_threshold = 50, return_filled_mask=True):
#Performs complete mammogram segmentation. Rotates image if needed, identifies 
#and segments breast tissue and eliminates the pectoral muscle and other labels. 
#Then resizes image and expands borders to desired shape. The expanded borders 
#beyond the resized image are filled with zeros. 
#If the resized image shape would exceed the desired expanded shape, image is not 
#resized. 
#Additionally, this function returns the mask binarized (and flipped when 
#necessary). The mask contour can also be filled (enabled by default). 
#Default parameters are set for my specific DDSM dataset.
#Parameters:
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
    segmented_breast                  = largest_contour_segmenting_based_in_pixel_luminancy_threshold(mammo_flipped, contour_threshold, width_darken, height_darken)
    segmented_breast_without_pectoral = expand_image_borders(crop_pectoral(segmented_breast, threshold=pectoral_threshold, pectoral_angle=pectoral_angle), expanded_height=expanded_height, expanded_width=expanded_width, resize_percentage=resize_percentage)    
    binarized_mask = expand_image_borders(array_binarize_with_threshold(mask_flipped, mask_threshold), expanded_height=expanded_height, expanded_width=expanded_width, resize_percentage=resize_percentage)

    if return_filled_mask==True:
        return (segmented_breast_without_pectoral, fill_mask(binarized_mask))
    else:
        return (segmented_breast_without_pectoral, binarized_mask)

################################################################################

def fill_mask(mask_bin):
#Returns the largest contour of the input mask filled.

    try:
        [contours, contour_hierarchy] = cv2.findContours(mask_bin, mode=cv2.RETR_CCOMP , method=cv2.CHAIN_APPROX_NONE)  
        # Get the largest contour index.
        largest_contour_index = mp.get_largest_contour_index(contours)
        # Generate a black base image for contour with the same shape as the passed image.
        largest_contour_mask = mp.array_generate_black_background(mask_bin)  
        cv2.drawContours(image=largest_contour_mask, contours=[contours[largest_contour_index]], contourIdx=-1, color=(255), thickness=-1)
        return largest_contour_mask
    except:
        return mask_bin

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