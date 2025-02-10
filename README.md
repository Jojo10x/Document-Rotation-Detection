# Document Rotation Detection

This project implements a Python-based solution to detect and correct the rotation of a document image using OpenCV. It leverages edge detection and the Hough transform to determine the skew angle of the document and then corrects it by rotating the image.

## Features

- Detects the rotation angle of a document
- Corrects the document orientation by rotating it to the proper alignment
- Saves the corrected image to the specified location
- Handles both grayscale and color images

## Algorithms

1. **Canny Edge Detection**: To extract the edges of the document.
2. **Hough Line Transform**: To detect lines and calculate the skew angle.
3. **Image Rotation**: To rotate the image based on the detected angle.
