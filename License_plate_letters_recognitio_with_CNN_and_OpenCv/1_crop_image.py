import cv2
import numpy as np
 
cropping = False
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0
 
image = cv2.imread('1_Input_images_for_cropping/license_plate_1.jpg')
oriImage = image.copy()
 
 
def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped image", roi)
            cv2.imwrite('2_Croped_image/Cropped_image.jpg',roi)
 
cv2.namedWindow("Mouse select cropping area")
cv2.setMouseCallback("Mouse select cropping area", mouse_crop)
 
while True:
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("Mouse select cropping area", image)
 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("Mouse select cropping area", i)
 
    k = cv2.waitKey(1)
    if k == 27:         # wait for ESC key to exit
        break
 
# close all open windows
cv2.destroyAllWindows()