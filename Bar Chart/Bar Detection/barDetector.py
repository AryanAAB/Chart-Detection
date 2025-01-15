import cv2
import numpy as np
import os

def extend_line(x1, y1, x2, y2, img_width, img_height):
    if x1 == x2:  # Vertical line
        return x1, 0, x2, img_height
    elif y1 == y2:  # Horizontal line
        return 0, y1, img_width, y2
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x_start = 0
        y_start = int(intercept)
        x_end = img_width
        y_end = int(slope * x_end + intercept)
        return x_start, y_start, x_end, y_end

def get_image(filename:str):
    if filename is None or not isinstance(filename, str):
        return None
    
    image = cv2.imread(filename)

    if image is None:
        return None
    
    img_width, img_height = 1000, 1000

    return cv2.resize(image, (img_width, img_height))

def detect_axis(filename:str):
    image = get_image(filename) 

    if image is None:
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)

    leftmost_line = None
    bottommost_line = None
    img_height, img_width = gray.shape

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if abs(x1 - x2) < 10:  # Approx. vertical
                if leftmost_line is None or x1 < leftmost_line[0]:
                    leftmost_line = (x1, y1, x2, y2)
            
            elif abs(y1 - y2) < 10:  # Approx. horizontal
                if bottommost_line is None or y1 > bottommost_line[1]:
                    bottommost_line = (x1, y1, x2, y2)

    # Extend lines to meet at a common point
    if leftmost_line and bottommost_line:
        lx1, ly1, lx2, ly2 = extend_line(*leftmost_line, img_width, img_height)
        bx1, by1, bx2, by2 = extend_line(*bottommost_line, img_width, img_height)

        return (lx1, ly1, lx2, ly2), (bx1, by1, bx2, by2)

    return leftmost_line, bottommost_line

def find_bar(filename:str, drawBinary:bool=False)->cv2.Mat:
    image = get_image(filename)

    if image is None:
        return None

    # Convert to HSV for color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply Gaussian Blur to reduce noise
    hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
    gray_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get binary image and then threshold this image as well
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(binary_image, 100, 255, cv2.THRESH_BINARY_INV)

    if draw_binary:
        cv2.imshow("HSV Image", hsv_image)
        cv2.imshow("Binary Image", binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and filter filled boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        bounding_box_area = w * h
        
        # Check if contour is a filled box
        if contour_area / bounding_box_area > 0.87:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 7)
    
    return image

def process_images_in_directory(directory:str, drawBinary:bool=False):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            
            processed_image = find_bar(file_path, drawBinary)
            x_axis, y_axis = detect_axis(file_path)
            
            if processed_image is not None:

                if x_axis is not None:
                    cv2.line(processed_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (255, 0, 0), 5)
                if y_axis is not None:
                    cv2.line(processed_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 5)

                cv2.imshow(f"Processed Image - {filename}", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Error processing image: {file_path}")

if __name__ == '__main__':

    outliers = False

    if outliers: 
        images_folder = 'Dataset/outliers'
        draw_binary = True
    else:
        images_folder = 'Dataset/images'
        draw_binary = True

    process_images_in_directory(images_folder, draw_binary)
