import cv2
import numpy as np

def im_diff(img1: np.ndarray, img2: np.ndarray, thresh: int, is_top_camera: bool) -> tuple:
    """
    Computes the absolute difference between two images, applies morphological 
    operations to reduce noise, and finds the largest contour.

    Args:
        img1 (np.ndarray): The current frame.
        img2 (np.ndarray): The background reference frame.
        thresh (int): Threshold value for binary conversion.
        is_top_camera (bool): Flag to adjust kernel sizes based on perspective.

    Returns:
        tuple: (processed_mask, largest_contour)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Blur to reduce minor lighting noise before differencing
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Find the difference and threshold it into a binary image
    diff = cv2.absdiff(blur1, blur2)
    _, blacked = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY) 

    # Adjust kernel sizes based on camera perspective
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    if is_top_camera:     
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    else:
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Morphological operations to fill holes and remove small noise
    dilated = cv2.dilate(blacked, kernel_large, iterations=2)
    processed_mask = cv2.erode(dilated, kernel_small, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_cont = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            largest_cont = contour
            max_area = area

    return (processed_mask, largest_cont)