import cv2
import numpy as np
from typing import Tuple

def detect_rotation_angle(image: np.ndarray) -> float:
    """
    Detect the rotation angle of a document using edge detection and Hough transform.
    
    Args:
        image: Input image as numpy array
    Returns:
        float: Detected rotation angle in degrees
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle > 90:
                angle = angle - 180
                
            if -60 <= angle <= 60:
                angles.append(angle)
    
    if not angles:
        return 0.0
        
    return np.median(angles)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by the specified angle.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees
    Returns:
        np.ndarray: Rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated

def process_document(image_path: str) -> Tuple[np.ndarray, float]:
    """
    Main function to process the document image.
    
    Args:
        image_path: Path to the input image
    Returns:
        Tuple[np.ndarray, float]: Corrected image and detected angle
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    angle = detect_rotation_angle(image)
    
    corrected_image = rotate_image(image, -angle)  
    
    return corrected_image, angle